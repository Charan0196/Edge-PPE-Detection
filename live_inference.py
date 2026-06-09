#!/usr/bin/env python3
"""Run live ONNX inference with required real-time metrics overlay."""

from __future__ import annotations

import argparse
import ast
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="Quantized ONNX model path.")
    parser.add_argument("--source", default="0", help="Webcam index or video file path.")
    parser.add_argument("--data", type=Path, help="YOLO data.yaml with class names.")
    parser.add_argument("--imgsz", default=416, type=int)
    parser.add_argument("--conf", default=0.25, type=float)
    parser.add_argument("--iou", default=0.45, type=float)
    parser.add_argument("--providers", nargs="*", default=None, help="Optional ONNX Runtime providers.")
    return parser.parse_args()


def load_names(data_yaml: Path | None, model_path: Path) -> list[str]:
    if data_yaml and data_yaml.exists():
        data = yaml.safe_load(data_yaml.read_text())
        names = data.get("names", {})
        if isinstance(names, dict):
            return [names[index] for index in sorted(names)]
        if isinstance(names, list):
            return names

    try:
        model = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        for item in model.get_modelmeta().custom_metadata_map.values():
            if "helmet" in item or "person" in item:
                parsed = ast.literal_eval(item)
                if isinstance(parsed, dict):
                    return [parsed[index] for index in sorted(parsed)]
    except Exception:
        pass

    return ["class_0", "class_1", "class_2"]


def letterbox(image: np.ndarray, new_shape: int) -> tuple[np.ndarray, float, tuple[float, float]]:
    height, width = image.shape[:2]
    scale = min(new_shape / height, new_shape / width)
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))
    pad_w = (new_shape - resized_width) / 2
    pad_h = (new_shape - resized_height) / 2

    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    top = int(round(pad_h - 0.1))
    bottom = int(round(pad_h + 0.1))
    left = int(round(pad_w - 0.1))
    right = int(round(pad_w + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded, scale, (pad_w, pad_h)


def preprocess(frame: np.ndarray, imgsz: int) -> tuple[np.ndarray, float, tuple[float, float]]:
    padded, scale, pad = letterbox(frame, imgsz)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[None]
    return np.ascontiguousarray(tensor), scale, pad


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    converted = np.empty_like(boxes)
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return converted


def postprocess(
    output: np.ndarray,
    frame_shape: tuple[int, int],
    scale: float,
    pad: tuple[float, float],
    conf_threshold: float,
    iou_threshold: float,
) -> list[tuple[list[int], int, float]]:
    prediction = np.squeeze(output)
    if prediction.ndim != 2:
        return []
    if prediction.shape[0] < prediction.shape[1] and prediction.shape[0] <= 128:
        prediction = prediction.T

    boxes = prediction[:, :4]
    class_scores = prediction[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    confidences = class_scores[np.arange(class_scores.shape[0]), class_ids]
    keep = confidences >= conf_threshold
    if not np.any(keep):
        return []

    boxes = xywh_to_xyxy(boxes[keep])
    confidences = confidences[keep]
    class_ids = class_ids[keep]

    pad_w, pad_h = pad
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / scale
    frame_h, frame_w = frame_shape
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, frame_w - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, frame_h - 1)

    nms_boxes = []
    for x1, y1, x2, y2 in boxes:
        nms_boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
    indices = cv2.dnn.NMSBoxes(nms_boxes, confidences.tolist(), conf_threshold, iou_threshold)

    detections: list[tuple[list[int], int, float]] = []
    if len(indices) == 0:
        return detections
    for index in np.array(indices).flatten():
        detections.append((nms_boxes[index], int(class_ids[index]), float(confidences[index])))
    return detections


def draw_overlay(
    frame: np.ndarray,
    detections: list[tuple[list[int], int, float]],
    names: list[str],
    preprocess_ms: float,
    inference_ms: float,
    postprocess_ms: float,
) -> None:
    colors = [(40, 170, 70), (255, 170, 40), (60, 80, 230), (200, 80, 180)]

    for box, class_id, confidence in detections:
        x, y, w, h = box
        color = colors[class_id % len(colors)]
        label = names[class_id] if class_id < len(names) else f"class_{class_id}"
        text = f"{label} {confidence:.2f}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x, max(0, y - text_size[1] - 8)), (x + text_size[0] + 8, y), color, -1)
        cv2.putText(frame, text, (x + 4, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    pipeline_ms = preprocess_ms + inference_ms + postprocess_ms
    fps = 1000.0 / pipeline_ms if pipeline_ms > 0 else 0.0
    metrics = [
        f"Inference FPS: {fps:.1f}",
        f"Preprocess: {preprocess_ms:.2f} ms",
        f"Inference: {inference_ms:.2f} ms",
        f"Post/NMS: {postprocess_ms:.2f} ms",
    ]
    y = 28
    for metric in metrics:
        cv2.putText(frame, metric, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(frame, metric, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 28


def open_source(source: str) -> cv2.VideoCapture:
    capture_source: int | str = int(source) if source.isdigit() else source
    capture = cv2.VideoCapture(capture_source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    return capture


def main() -> None:
    args = parse_args()
    providers = args.providers or ort.get_available_providers()
    session = ort.InferenceSession(str(args.model), providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    names = load_names(args.data, args.model)
    capture = open_source(args.source)

    print("Press q to quit.")
    while True:
        ok, frame = capture.read()
        if not ok:
            break

        t0 = time.perf_counter()
        tensor, scale, pad = preprocess(frame, args.imgsz)
        preprocess_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        output = session.run([output_name], {input_name: tensor})[0]
        inference_ms = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        detections = postprocess(output, frame.shape[:2], scale, pad, args.conf, args.iou)
        postprocess_ms = (time.perf_counter() - t2) * 1000

        draw_overlay(frame, detections, names, preprocess_ms, inference_ms, postprocess_ms)
        cv2.imshow("Edge PPE Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
