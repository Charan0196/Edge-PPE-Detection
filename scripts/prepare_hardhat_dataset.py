#!/usr/bin/env python3
"""Convert the Kaggle hardhat Pascal VOC dataset to YOLO format."""

from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml
from tqdm import tqdm


CLASSES = ["helmet", "person", "head"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True, type=Path, help="Directory containing Kaggle images and XML files.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output YOLO dataset directory.")
    parser.add_argument("--train-ratio", default=0.8, type=float)
    parser.add_argument("--val-ratio", default=0.1, type=float)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def find_image_for_xml(xml_path: Path, image_lookup: dict[str, Path]) -> Path | None:
    stem = xml_path.stem.lower()
    if stem in image_lookup:
        return image_lookup[stem]

    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return None

    filename = root.findtext("filename")
    if filename:
        return image_lookup.get(Path(filename).stem.lower())
    return None


def voc_to_yolo(xml_path: Path) -> list[str]:
    root = ET.parse(xml_path).getroot()
    width = int(float(root.findtext("size/width", "0")))
    height = int(float(root.findtext("size/height", "0")))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in {xml_path}")

    labels: list[str] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        if name not in CLASSES:
            continue

        box = obj.find("bndbox")
        if box is None:
            continue

        xmin = max(0.0, min(float(box.findtext("xmin", "0")), width))
        ymin = max(0.0, min(float(box.findtext("ymin", "0")), height))
        xmax = max(0.0, min(float(box.findtext("xmax", "0")), width))
        ymax = max(0.0, min(float(box.findtext("ymax", "0")), height))
        if xmax <= xmin or ymax <= ymin:
            continue

        x_center = ((xmin + xmax) / 2.0) / width
        y_center = ((ymin + ymax) / 2.0) / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height
        labels.append(f"{CLASSES.index(name)} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")

    return labels


def split_items(items: list[tuple[Path, Path]], train_ratio: float, val_ratio: float) -> dict[str, list[tuple[Path, Path]]]:
    train_end = int(len(items) * train_ratio)
    val_end = train_end + int(len(items) * val_ratio)
    return {"train": items[:train_end], "val": items[train_end:val_end], "test": items[val_end:]}


def main() -> None:
    args = parse_args()
    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("--train-ratio and --val-ratio must leave a non-empty test split.")

    images = [p for p in args.raw_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    xml_files = sorted(args.raw_dir.rglob("*.xml"))
    image_lookup = {p.stem.lower(): p for p in images}

    paired: list[tuple[Path, Path]] = []
    for xml_path in xml_files:
        image_path = find_image_for_xml(xml_path, image_lookup)
        if image_path is not None:
            paired.append((image_path, xml_path))

    if len(paired) < 500:
        raise RuntimeError(f"Found only {len(paired)} image/XML pairs. Check --raw-dir.")

    random.seed(args.seed)
    random.shuffle(paired)
    splits = split_items(paired, args.train_ratio, args.val_ratio)

    for split_name, split_items_ in splits.items():
        image_out = args.out_dir / "images" / split_name
        label_out = args.out_dir / "labels" / split_name
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)

        for image_path, xml_path in tqdm(split_items_, desc=f"Writing {split_name}"):
            labels = voc_to_yolo(xml_path)
            shutil.copy2(image_path, image_out / image_path.name)
            (label_out / f"{image_path.stem}.txt").write_text("\n".join(labels) + ("\n" if labels else ""))

    data_yaml = {
        "path": str(args.out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {idx: name for idx, name in enumerate(CLASSES)},
    }
    (args.out_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))

    print(f"Prepared {len(paired)} labeled images at {args.out_dir}")
    for split_name, split_items_ in splits.items():
        print(f"{split_name}: {len(split_items_)} images")


if __name__ == "__main__":
    main()
