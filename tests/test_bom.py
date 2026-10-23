# SPDX-FileCopyrightText: 2026 Jean-Pierre De Jesus DIAZ <me@jeandudey.tech>
#
# SPDX-License-Identifier: MIT

import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from scrfd.bom import Json, main, read_split
from scrfd.openimages import HUMAN_FACE, convert

HEADER = "ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf"
METADATA_HEADER = (
    "ImageID,Subset,OriginalURL,OriginalLandingURL,License,AuthorProfileURL,Author,Title,"
    "OriginalSize,OriginalMD5,Thumbnail300KURL,Rotation"
)
BY = "https://creativecommons.org/licenses/by/2.0/"
NC = "https://creativecommons.org/licenses/by-nc/2.0/"
RECIPE = {
    "variant": "scrfd_500m",
    "size": 640,
    "seed": 0,
    "lr": 0.01,
    "epochs": 640,
    "lr_steps": [440, 544],
    "warmup_iters": 1500,
    "batch_size": 16,
    "momentum": 0.9,
    "weight_decay": 5e-4,
}


def write_split(root: Path, split: str, ids: list[str]) -> None:
    images = root / "raw" / split
    images.mkdir(parents=True)
    rows = [f"{i},xclick,{HUMAN_FACE},1,0.1,0.5,0.2,0.6,0,0,0" for i in ids]
    metadata = [
        f"{i},{split},u,https://flickr.example/{i},{NC if i == 'nc' else BY},"
        f"https://flickr.example/people/{i},Author {i},Title {i},1,m,t,0"
        for i in ids
    ]
    for i in ids:
        cv2.imwrite(str(images / f"{i}.jpg"), np.zeros((100, 200, 3), np.uint8))
    (root / f"{split}.csv").write_text("\n".join([HEADER, *rows]) + "\n")
    (root / f"{split}-metadata.csv").write_text("\n".join([METADATA_HEADER, *metadata]) + "\n")
    convert(
        root / f"{split}.csv",
        images,
        root / "data",
        split,
        metadata_csv=root / f"{split}-metadata.csv",
    )


@pytest.fixture
def trained(tmp_path: Path) -> Path:
    write_split(tmp_path, "train", ["a", "b", "nc"])
    write_split(tmp_path, "val", ["c"])
    work = tmp_path / "work"
    work.mkdir()
    (work / "epoch_640.pth").write_bytes(b"weights")
    (work / "recipe.json").write_text(json.dumps(RECIPE))
    return tmp_path


def test_attribution(trained: Path) -> None:
    lines = (trained / "data" / "train" / "ATTRIBUTION.csv").read_text().splitlines()
    assert lines[0] == "ImageID,Title,Author,AuthorProfileURL,OriginalLandingURL,License"
    assert [line.split(",")[0] for line in lines[1:]] == ["a", "b"]  # "nc" is not CC BY
    assert "Author a" in lines[1]


def test_read_split_needs_a_credit_per_image(trained: Path) -> None:
    attribution = trained / "data" / "train" / "ATTRIBUTION.csv"
    kept = attribution.read_text().splitlines()[:2]
    attribution.write_text("\n".join(kept) + "\n")
    with pytest.raises(ValueError, match="does not credit 1 images"):
        read_split(trained / "data", "train", trained / "work")
    assert read_split(trained / "data", "val", trained / "work") is not None
    assert read_split(trained / "data", "test", trained / "work") is None


def run(trained: Path, *extra: str) -> tuple[Json, Json]:
    work = trained / "work"
    main(
        [
            "scrfd_500m",
            str(work / "epoch_640.pth"),
            "--author",
            "Jane Doe",
            "--data-root",
            str(trained / "data"),
            "--ap",
            "0.9",
            *extra,
        ]
    )
    return (
        json.loads((work / "bom.cdx.json").read_text()),
        json.loads((work / "bom.spdx3.json").read_text()),
    )


def test_boms(trained: Path) -> None:
    cdx, spdx = run(trained, "--license", "MIT")
    work = trained / "work"
    assert (work / "ATTRIBUTION.train.csv").read_bytes().count(b"\n") == 3
    assert (work / "ATTRIBUTION.val.csv").exists()

    model, train, val = cdx["components"]
    assert model["type"] == "machine-learning-model"
    assert model["hashes"][0]["content"] == hashlib.sha256(b"weights").hexdigest()
    refs = [d["ref"] for d in model["modelCard"]["modelParameters"]["datasets"]]
    assert refs == [train["bom-ref"], val["bom-ref"]]
    assert [lic["license"]["id"] for lic in train["licenses"]] == ["CC-BY-4.0", "CC-BY-2.0"]
    assert model["modelCard"]["quantitativeAnalysis"]["performanceMetrics"][0]["value"] == "0.9000"
    assert model["licenses"] == [{"expression": "MIT"}]

    graph = {e["spdxId"]: e for e in spdx["@graph"] if "spdxId" in e}
    kinds = [e["type"] for e in graph.values()]
    assert kinds.count("dataset_DatasetPackage") == 2
    relationships = [e for e in graph.values() if e["type"] == "Relationship"]
    trained_on = next(r for r in relationships if r["relationshipType"] == "trainedOn")
    tested_on = next(r for r in relationships if r["relationshipType"] == "testedOn")
    assert graph[trained_on["from"]]["type"] == "ai_AIPackage"
    assert graph[trained_on["to"][0]]["name"].endswith("(training)")
    assert graph[tested_on["to"][0]]["name"].endswith("(validation)")
    document = graph[next(i for i, e in graph.items() if e["type"] == "SpdxDocument")]
    assert set(document["element"]) == set(graph) - {document["spdxId"]}


def test_without_validation_split(trained: Path) -> None:
    (trained / "data" / "val" / "labelv2.txt").unlink()
    cdx, spdx = run(trained)
    assert len(cdx["components"]) == 2
    assert all(e.get("relationshipType") != "testedOn" for e in spdx["@graph"])
    assert "licenses" not in cdx["components"][0]


@pytest.mark.skipif("BOM_SCHEMAS" not in os.environ, reason="BOM_SCHEMAS is not set")
def test_boms_match_the_schemas(trained: Path) -> None:
    """Validate against the CycloneDX 1.6 and SPDX 3.0.1 JSON schemas found in ``BOM_SCHEMAS``."""
    import jsonschema
    from referencing import Registry, Resource

    folder = Path(os.environ["BOM_SCHEMAS"])

    def load(name: str) -> Json:
        return json.loads((folder / name).read_text())

    registry = Registry().with_resources(
        [
            (f"http://cyclonedx.org/schema/{name}", Resource.from_contents(load(name)))
            for name in ("spdx.schema.json", "jsf-0.82.schema.json")
        ]
    )
    cdx, spdx = run(trained, "--license", "MIT")
    jsonschema.Draft7Validator(load("bom-1.6.schema.json"), registry=registry).validate(cdx)
    jsonschema.Draft202012Validator(load("spdx-schema.json")).validate(spdx)
