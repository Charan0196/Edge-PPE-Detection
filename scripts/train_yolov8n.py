#!/usr/bin/env python3
"""Train a YOLOv8n FP32 hardhat detector."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="YOLO data.yaml path.")
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--imgsz", default=416, type=int)
    parser.add_argument("--batch", default=16, type=int)
    parser.add_argument("--device", default=None, help="Device passed to Ultralytics, for example 0, cpu, or mps.")
    parser.add_argument("--project", default="runs/hardhat")
    parser.add_argument("--name", default="yolov8n_fp32")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO("yolov8n.pt")
    train_kwargs = {
        "data": str(args.data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": str(Path(args.project).resolve()),
        "name": args.name,
        "pretrained": True,
        "amp": False,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device

    model.train(**train_kwargs)
    print("Training complete. Running final validation...")
    model.val(data=str(args.data), imgsz=args.imgsz, split="test")


if __name__ == "__main__":
    main()
