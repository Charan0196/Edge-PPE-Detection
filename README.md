# Edge PPE Detection: Hardhat Compliance

This project implements the assignment pipeline for a real-world industrial safety problem: detecting people, hardhats, and uncovered heads in construction-site imagery so an edge camera can flag missing PPE.

## Dataset

- Dataset: [Safety Helmet Detection / Hard Hat Dataset on Kaggle](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection)
- Size: 5,000 images
- Labels: `helmet`, `person`, `head`
- Annotation format: Pascal VOC XML, converted to YOLO format by `scripts/prepare_hardhat_dataset.py`
- License: CC0 / Public Domain on Kaggle

## Model Choice

- Baseline: YOLOv8n trained in FP32
- Edge format: ONNX
- Quantization: FP16 weights through ONNX float16 conversion

YOLOv8n is small enough for edge-class hardware while still giving practical object detection quality. FP16 ONNX was selected because it usually gives a large model-size reduction with minimal accuracy loss and does not require an INT8 calibration set.

## Repository Layout

```text
.
├── ASSIGNMENT_SUBMISSION.md
├── live_inference.py
├── notebooks/
│   └── train_baseline_yolov8n.ipynb
├── requirements.txt
├── scripts/
│   ├── benchmark_models.py
│   ├── convert_to_onnx_fp16.py
│   ├── prepare_hardhat_dataset.py
│   └── train_yolov8n.py
└── weights/
    └── .gitkeep
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the Kaggle dataset into `data/raw/hard-hat-detection/` so the directory contains the images and XML annotation files from the dataset.

With a configured Kaggle API token, one way to do that is:

```bash
mkdir -p data/raw/hard-hat-detection
kaggle datasets download -d andrewmvd/hard-hat-detection \
  -p data/raw/hard-hat-detection \
  --unzip
```

## Training

Convert Pascal VOC annotations to YOLO:

```bash
python scripts/prepare_hardhat_dataset.py \
  --raw-dir data/raw/hard-hat-detection \
  --out-dir data/hardhat_yolo \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --seed 42
```

Train the FP32 YOLOv8n baseline:

```bash
python scripts/train_yolov8n.py \
  --data data/hardhat_yolo/data.yaml \
  --epochs 50 \
  --imgsz 416 \
  --batch 16 \
  --project runs/hardhat \
  --name yolov8n_fp32
```

Copy the resulting model for submission:

```bash
cp runs/hardhat/yolov8n_fp32/weights/best.pt weights/yolov8n_hardhat_fp32.pt
```

## ONNX FP16 Conversion

```bash
python scripts/convert_to_onnx_fp16.py \
  --weights weights/yolov8n_hardhat_fp32.pt \
  --output weights/yolov8n_hardhat_fp16.onnx \
  --imgsz 416
```

## Live Inference

Run on webcam:

```bash
python live_inference.py \
  --model weights/yolov8n_hardhat_fp16.onnx \
  --data data/hardhat_yolo/data.yaml \
  --source 0 \
  --imgsz 416 \
  --conf 0.25 \
  --iou 0.45
```

Run on a test video:

```bash
python live_inference.py \
  --model weights/yolov8n_hardhat_fp16.onnx \
  --data data/hardhat_yolo/data.yaml \
  --source path/to/test_video.mp4
```

The video overlay includes:

- Bounding boxes with class labels and confidence scores
- Live inference FPS excluding rendering time
- Pre-processing latency in milliseconds
- Post-processing/NMS latency in milliseconds

## Benchmark

After training and conversion, run:

```bash
python scripts/benchmark_models.py \
  --data data/hardhat_yolo/data.yaml \
  --fp32 weights/yolov8n_hardhat_fp32.pt \
  --edge weights/yolov8n_hardhat_fp16.onnx \
  --source data/hardhat_yolo/images/test \
  --imgsz 416
```

Benchmark collected locally on Apple M1 CPU using `imgsz=416`, `warmup=3`, and `runs=30`.

| Metric | FP32 YOLOv8n `.pt` | FP16 ONNX Edge Model | Change |
|---|---:|---:|---:|
| Model size (MB) | 23.29 | 5.83 | -17.46 |
| mAP50-95 | 0.3125 | 0.3037 | -0.0088 |
| FPS on local machine | 5.63 | 6.24 | +0.60 |

## Required Links

- GitHub repository: `TODO: paste public repo URL`
- FP32 weights: `TODO: paste Google Drive / Hugging Face URL`
- FP16 ONNX weights: `TODO: paste Google Drive / Hugging Face URL`
- 2-3 minute video proof: `TODO: paste unlisted YouTube / Loom / Drive URL`
# Edge-PPE-Detection
