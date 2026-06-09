#!/usr/bin/env python3
"""Export YOLOv8 weights to ONNX and convert internal weights to FP16."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import onnx
from onnxconverter_common import float16
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path, help="Trained FP32 .pt weights.")
    parser.add_argument("--output", required=True, type=Path, help="Output FP16 ONNX path.")
    parser.add_argument("--imgsz", default=416, type=int)
    parser.add_argument("--opset", default=12, type=int)
    parser.add_argument("--dynamic", action="store_true", help="Export ONNX with dynamic input shape.")
    parser.add_argument("--simplify", action="store_true", help="Ask Ultralytics to simplify the exported ONNX graph.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))
    exported = Path(
        model.export(
            format="onnx",
            imgsz=args.imgsz,
            opset=args.opset,
            dynamic=args.dynamic,
            simplify=args.simplify,
            half=False,
        )
    )

    fp32_onnx = args.output.with_name(f"{args.output.stem}_fp32.onnx")
    shutil.copy2(exported, fp32_onnx)

    onnx_model = onnx.load(fp32_onnx)
    fp16_model = float16.convert_float_to_float16(onnx_model, keep_io_types=True)
    onnx.checker.check_model(fp16_model)
    onnx.save(fp16_model, args.output)

    print(f"FP32 ONNX: {fp32_onnx}")
    print(f"FP16 ONNX: {args.output}")


if __name__ == "__main__":
    main()
