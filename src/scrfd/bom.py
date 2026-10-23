# SPDX-FileCopyrightText: 2026 Jean-Pierre De Jesus DIAZ <me@jeandudey.tech>
#
# SPDX-License-Identifier: MIT

"""``scrfd-bom``: describe a model trained on Open Images as a CycloneDX and an SPDX 3.0 BOM.

Both BOMs name the model (with its SHA-256), the datasets it was trained and validated on with
their licenses, and the training recipe. The per-image credits that CC BY asks for are copied
next to them as ``ATTRIBUTION.<split>.csv``. Run ``scrfd-train`` and
``scrfd-convert-openimages --metadata`` first.
"""

import argparse
import csv
import hashlib
import json
import shutil
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch

from .dataset import load_records
from .variants import CONFIGS

Json = dict[str, Any]

DATASET_NAME = "Open Images V7"
DATASET_HOME = "https://storage.googleapis.com/openimages/web/index.html"
DATASET_PAPER = "https://arxiv.org/abs/1811.00982"
ANNOTATION_LICENSE = "CC-BY-4.0"
IMAGE_LICENSE = "CC-BY-2.0"
DATASET_LICENSES = f"{ANNOTATION_LICENSE} AND {IMAGE_LICENSE}"
ROLES = {"train": "training", "val": "validation"}
HYPERPARAMETERS = (
    "lr",
    "epochs",
    "lr_steps",
    "warmup_iters",
    "batch_size",
    "momentum",
    "weight_decay",
    "size",
    "seed",
)
INPUT_FORMAT = "RGB image batch (3, H, W), normalised as (x - 127.5) / 128, H and W multiples of 32"
LIMITATION = (
    "Trained on photographs from Flickr as released in Open Images; accuracy on small, blurred or "
    "crowded faces and the demographic coverage have not been assessed."
)
SENSITIVE = "Photographs of human faces (no identities or landmarks are used)"


@dataclass(frozen=True)
class Split:
    """One split of the converted dataset, with the credit file that goes with it."""

    name: str
    role: str
    images: int
    faces: int
    size: int | None  # bytes of the image files, if they are available
    attribution: str  # file name next to the BOMs
    attribution_sha256: str


@dataclass(frozen=True)
class Provenance:
    """What the two BOMs state about a model."""

    variant: str
    author: str
    model_sha256: str
    hyperparameters: dict[str, str]
    splits: tuple[Split, ...]
    created: datetime
    ap: float | None = None
    license: str | None = None
    scrfd_version: str = "unknown"
    torch_version: str = torch.__version__

    @property
    def timestamp(self) -> str:
        return self.created.strftime("%Y-%m-%dT%H:%M:%SZ")

    @property
    def uses_kps(self) -> bool:
        return CONFIGS[self.variant].use_kps

    def training(self) -> tuple[Split, ...]:
        return tuple(s for s in self.splits if s.role == "training")

    def validation(self) -> tuple[Split, ...]:
        return tuple(s for s in self.splits if s.role == "validation")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def modifications(size: str) -> list[str]:
    return [
        "kept only the 'Human face' boxes (/m/0dzct)",
        "dropped the images with an IsGroupOf face box",
        "converted the boxes to pixel coordinates in the RetinaFace annotation format",
        f"training augmentation: random square crop, resize to {size}x{size}, horizontal flip, "
        "photometric distortion",
    ]


def attribution_text(split: Split) -> str:
    return (
        f"{DATASET_NAME} annotations (c) Google LLC, licensed {ANNOTATION_LICENSE}. Images (c) "
        f"their authors, licensed {IMAGE_LICENSE}; the credit of each is in {split.attribution}."
    )


def read_split(data_root: Path, name: str, output: Path) -> Split | None:
    """Describe ``data_root/<name>`` and copy its credits to ``output``; ``None`` if it is absent.

    Every image of the split has to be credited.
    """
    folder = data_root / name
    if not (folder / "labelv2.txt").exists():
        return None
    records = load_records(folder / "labelv2.txt")
    source = folder / "ATTRIBUTION.csv"
    if not source.exists():
        msg = f"{source} is missing: run scrfd-convert-openimages with --metadata"
        raise FileNotFoundError(msg)
    with source.open(newline="") as file:
        credited = {row["ImageID"] for row in csv.DictReader(file)}
    uncredited = {Path(r.name).stem for r in records} - credited
    if uncredited:
        msg = f"{source} does not credit {len(uncredited)} images, e.g. {min(uncredited)}"
        raise ValueError(msg)
    paths = [folder / "images" / r.name for r in records]
    size = sum(p.stat().st_size for p in paths) if all(p.exists() for p in paths) else None
    target = output / f"ATTRIBUTION.{name}.csv"
    shutil.copyfile(source, target)
    return Split(
        name=name,
        role=ROLES[name],
        images=len(records),
        faces=sum(len(r.boxes) for r in records),
        size=size,
        attribution=target.name,
        attribution_sha256=sha256(target),
    )


def cyclonedx_dataset(split: Split, provenance: Provenance) -> Json:
    description = (
        f"{DATASET_NAME} faces, {split.role} split: {split.images} images, {split.faces} faces. "
        f"Modifications: {'; '.join(modifications(provenance.hyperparameters['size']))}."
    )
    return {
        "type": "data",
        "bom-ref": f"dataset-{split.name}",
        "name": f"{DATASET_NAME} faces ({split.role})",
        "version": "V7",
        "description": description,
        "copyright": attribution_text(split),
        "licenses": [
            {"license": {"id": ANNOTATION_LICENSE}},
            {"license": {"id": IMAGE_LICENSE}},
        ],
        "externalReferences": [
            {"type": "website", "url": DATASET_HOME},
            {"type": "documentation", "url": DATASET_PAPER},
        ],
        "properties": [
            {"name": "scrfd:images", "value": str(split.images)},
            {"name": "scrfd:faces", "value": str(split.faces)},
            {"name": "scrfd:attribution-file", "value": split.attribution},
            {"name": "scrfd:attribution-sha256", "value": split.attribution_sha256},
        ],
        "data": [
            {
                "type": "dataset",
                "name": f"{DATASET_NAME} faces ({split.role})",
                "contents": {"url": DATASET_HOME},
                "classification": "public",
                "sensitiveData": [SENSITIVE],
                "description": description,
                "governance": {"custodians": [{"organization": {"name": "Google LLC"}}]},
            }
        ],
    }


def cyclonedx(provenance: Provenance) -> Json:
    """A CycloneDX 1.6 ML-BOM."""
    outputs = "Per stride: classification scores, box distances in units of the stride"
    if provenance.uses_kps:
        outputs += ", keypoint distances"
    card: Json = {
        "modelParameters": {
            "approach": {"type": "supervised"},
            "task": "face detection",
            "architectureFamily": "SCRFD",
            "modelArchitecture": provenance.variant,
            "datasets": [{"ref": f"dataset-{s.name}"} for s in provenance.splits],
            "inputs": [{"format": INPUT_FORMAT}],
            "outputs": [{"format": outputs}],
        },
        "considerations": {
            "useCases": ["Detecting faces in images"],
            "technicalLimitations": [LIMITATION],
            "ethicalConsiderations": [
                {
                    "name": "The training data contains photographs of people's faces",
                    "mitigationStrategy": "Only CC BY-licensed images released in Open Images "
                    "are used, credited in the ATTRIBUTION files; no identities are used.",
                }
            ],
        },
        "properties": [
            {"name": f"scrfd:{key}", "value": value}
            for key, value in provenance.hyperparameters.items()
        ]
        + [{"name": "scrfd:torch-version", "value": provenance.torch_version}],
    }
    if provenance.ap is not None:
        card["quantitativeAnalysis"] = {
            "performanceMetrics": [
                {
                    "type": "AP@0.5",
                    "value": f"{provenance.ap:.4f}",
                    "slice": f"{DATASET_NAME} faces, validation split",
                }
            ]
        }
    model: Json = {
        "type": "machine-learning-model",
        "bom-ref": "model",
        "name": provenance.variant,
        "description": f"{provenance.variant} face detector trained with scrfd",
        "authors": [{"name": provenance.author}],
        "hashes": [{"alg": "SHA-256", "content": provenance.model_sha256}],
        "modelCard": card,
    }
    if provenance.license is not None:
        model["licenses"] = [{"expression": provenance.license}]
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "serialNumber": "urn:uuid:"
        + str(uuid.uuid5(uuid.NAMESPACE_URL, f"{provenance.model_sha256}/{provenance.timestamp}")),
        "version": 1,
        "metadata": {
            "timestamp": provenance.timestamp,
            "tools": {
                "components": [
                    {
                        "type": "application",
                        "name": "scrfd-bom",
                        "version": provenance.scrfd_version,
                    }
                ]
            },
            "authors": [{"name": provenance.author}],
        },
        "components": [model, *(cyclonedx_dataset(s, provenance) for s in provenance.splits)],
    }


def spdx(provenance: Provenance) -> Json:
    """An SPDX 3.0.1 document with the AI and Dataset profiles, as JSON-LD."""
    base = f"urn:scrfd-bom:{provenance.model_sha256[:16]}"
    info = "_:creationInfo"
    graph: list[Json] = []
    relationships = 0

    def element(kind: str, slug: str, **fields: object) -> str:
        spdx_id = f"{base}/{slug}"
        graph.append({"type": kind, "spdxId": spdx_id, "creationInfo": info, **fields})
        return spdx_id

    def relate(source: str, kind: str, targets: list[str]) -> None:
        nonlocal relationships
        relationships += 1
        fields = {"from": source, "relationshipType": kind, "to": targets}
        element("Relationship", f"relationship-{relationships}", **fields)

    def expression(slug: str, text: str) -> str:
        return element(
            "simplelicensing_LicenseExpression", slug, simplelicensing_licenseExpression=text
        )

    author = element("Person", "author", name=provenance.author)
    tool = element("Tool", "tool", name=f"scrfd-bom {provenance.scrfd_version}")
    document_license = expression("license-document", "CC0-1.0")
    dataset_license = expression("license-dataset", DATASET_LICENSES)

    size = provenance.hyperparameters["size"]
    datasets: dict[str, str] = {}
    for split in provenance.splits:
        credits = element(
            "software_File",
            f"attribution-{split.name}",
            name=split.attribution,
            contentType="text/csv",
            software_primaryPurpose="documentation",
            verifiedUsing=[
                {"type": "Hash", "algorithm": "sha256", "hashValue": split.attribution_sha256}
            ],
        )
        fields: Json = {
            "name": f"{DATASET_NAME} faces ({split.role})",
            "summary": f"{split.images} images, {split.faces} faces",
            "software_primaryPurpose": "data",
            "software_packageVersion": "V7",
            "software_homePage": DATASET_HOME,
            "software_downloadLocation": DATASET_HOME,
            "software_attributionText": [attribution_text(split)],
            "software_copyrightText": attribution_text(split),
            "dataset_datasetType": ["image"],
            "dataset_dataCollectionProcess": (
                f"Photographs from Flickr, annotated by Google; see {DATASET_PAPER}"
            ),
            "dataset_dataPreprocessing": modifications(size),
            "dataset_intendedUse": f"Face detection, {split.role}",
            "dataset_hasSensitivePersonalInformation": "yes",
            "dataset_datasetAvailability": "directDownload",
            "dataset_knownBias": ["Demographic coverage of the photographs has not been assessed"],
        }
        if split.size is not None:
            fields["dataset_datasetSize"] = split.size
        datasets[split.name] = element("dataset_DatasetPackage", f"dataset-{split.name}", **fields)
        relate(datasets[split.name], "contains", [credits])
        relate(datasets[split.name], "hasDeclaredLicense", [dataset_license])

    hyperparameters = [
        {"type": "DictionaryEntry", "key": key, "value": value}
        for key, value in provenance.hyperparameters.items()
    ]
    model_fields: Json = {
        "name": provenance.variant,
        "summary": f"{provenance.variant} face detector trained with scrfd",
        "originatedBy": [author],
        "builtTime": provenance.timestamp,
        "software_primaryPurpose": "model",
        "verifiedUsing": [
            {"type": "Hash", "algorithm": "sha256", "hashValue": provenance.model_sha256}
        ],
        "ai_autonomyType": "no",
        "ai_domain": ["computer vision"],
        "ai_typeOfModel": ["SCRFD convolutional face detector"],
        "ai_hyperparameter": hyperparameters,
        "ai_informationAboutTraining": (
            "Trained from scratch with scrfd-train on the training split(s) listed as trainedOn, "
            f"with PyTorch {provenance.torch_version}"
        ),
        "ai_informationAboutApplication": INPUT_FORMAT,
        "ai_modelDataPreprocessing": modifications(size),
        "ai_limitation": LIMITATION,
        "ai_useSensitivePersonalInformation": "yes",
    }
    if provenance.ap is not None:
        model_fields["ai_metric"] = [
            {"type": "DictionaryEntry", "key": "AP@0.5", "value": f"{provenance.ap:.4f}"}
        ]
    model = element("ai_AIPackage", "model", **model_fields)
    relate(model, "trainedOn", [datasets[s.name] for s in provenance.training()])
    if provenance.validation():
        relate(model, "testedOn", [datasets[s.name] for s in provenance.validation()])
    if provenance.license is not None:
        relate(model, "hasDeclaredLicense", [expression("license-model", provenance.license)])

    members = [e["spdxId"] for e in graph]
    element(
        "SpdxDocument",
        "document",
        name=f"{provenance.variant} ML-BOM",
        dataLicense=document_license,
        profileConformance=["core", "software", "ai", "dataset", "simpleLicensing"],
        rootElement=[model],
        element=members,
    )
    creation_info = {
        "type": "CreationInfo",
        "@id": info,
        "specVersion": "3.0.1",
        "created": provenance.timestamp,
        "createdBy": [author],
        "createdUsing": [tool],
    }
    return {
        "@context": "https://spdx.org/rdf/3.0.1/spdx-context.jsonld",
        "@graph": [creation_info, *graph],
    }


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="scrfd-bom", description=__doc__)
    parser.add_argument("variant", choices=sorted(CONFIGS), help="model variant")
    parser.add_argument("snapshot", type=Path, help="the trained model, e.g. epoch_640.pth")
    parser.add_argument("--author", required=True, help="who trained and publishes the model")
    parser.add_argument("--data-root", type=Path, default=Path("data/openimages"))
    parser.add_argument(
        "--recipe", type=Path, help="recipe.json of scrfd-train (default: next to the snapshot)"
    )
    parser.add_argument(
        "--output", type=Path, help="directory for the BOMs (default: next to the snapshot)"
    )
    parser.add_argument("--ap", type=float, help="AP@0.5 on the validation split, to record")
    parser.add_argument("--license", help="SPDX license expression of the model weights")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output = args.output or args.snapshot.parent
    recipe = json.loads((args.recipe or args.snapshot.parent / "recipe.json").read_text())
    if recipe["variant"] != args.variant:
        msg = f"the recipe is of {recipe['variant']}, not {args.variant}"
        raise SystemExit(msg)
    output.mkdir(parents=True, exist_ok=True)
    splits = tuple(s for name in ROLES if (s := read_split(args.data_root, name, output)))
    if not any(s.role == "training" for s in splits):
        msg = f"{args.data_root}/train/labelv2.txt not found"
        raise SystemExit(msg)
    try:
        scrfd_version = version("scrfd")
    except PackageNotFoundError:
        scrfd_version = "unknown"
    provenance = Provenance(
        variant=args.variant,
        author=args.author,
        model_sha256=sha256(args.snapshot),
        hyperparameters={key: json.dumps(recipe[key]) for key in HYPERPARAMETERS},
        splits=splits,
        created=datetime.now(UTC).replace(microsecond=0),
        ap=args.ap,
        license=args.license,
        scrfd_version=scrfd_version,
    )
    for name, bom in (
        ("bom.cdx.json", cyclonedx(provenance)),
        ("bom.spdx3.json", spdx(provenance)),
    ):
        (output / name).write_text(json.dumps(bom, indent=2) + "\n")
        print(output / name)


if __name__ == "__main__":
    main()
