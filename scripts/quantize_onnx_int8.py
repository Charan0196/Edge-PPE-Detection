#!/usr/bin/env python3
"""Quantize an ONNX model to INT8 using ONNX Runtime dynamic quantization."""

from __future__ import annotations

import argparse
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input ONNX model path.")
    parser.add_argument("--output", required=True, type=Path, help="Output INT8 ONNX model path.")
    parser.add_argument("--per-channel", action="store_true", help="Enable per-channel quantization for weights.")
    parser.add_argument("--reduce-range", action="store_true", help="Use reduced range for quantized values.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(args.input),
        model_output=str(args.output),
        weight_type=QuantType.QInt8,
        per_channel=args.per_channel,
        reduce_range=args.reduce_range,
    )
    print(f"INT8 model saved to {args.output}")


if __name__ == "__main__":
    main()
