# Edge PPE Detection Assignment Submission

## Phase 1: Data Sourcing And Base Training

Problem selected: industrial PPE compliance monitoring for construction sites. The detector identifies `helmet`, `person`, and uncovered `head` classes so an edge camera can help detect missing hardhats.

Dataset used: [Safety Helmet Detection / Hard Hat Dataset on Kaggle](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection). The dataset contains 5,000 images with Pascal VOC bounding-box annotations for `helmet`, `person`, and `head`, satisfying the minimum 500-image requirement.

Baseline model: YOLOv8n trained in FP32 precision with Ultralytics.

Training entry points:

- `notebooks/train_baseline_yolov8n.ipynb`
- `scripts/prepare_hardhat_dataset.py`
- `scripts/train_yolov8n.py`

## Phase 2: Edge Conversion And Quantization

Edge format selected: ONNX.

Quantization selected: FP16.

Rationale: FP16 conversion cuts the model footprint substantially while preserving most of the FP32 accuracy. It is simpler and safer than INT8 for this assignment because INT8 requires calibration data and can cause larger accuracy degradation if the calibration set does not represent the deployment environment well.

Conversion entry point:

- `scripts/convert_to_onnx_fp16.py`

## Phase 3: Inference Script And Metrics

Live inference entry point:

- `live_inference.py`

The script loads the ONNX FP16 model with ONNX Runtime and overlays the required real-time metrics:

- Bounding boxes with class names and confidence scores
- Inference FPS excluding rendering time
- Pre-processing latency in milliseconds
- Post-processing/NMS latency in milliseconds

## Source Code And Model Weights

- Public GitHub repository: `https://github.com/Charan0196/Edge-PPE-Detection`
- Original FP32 model weights: `weights/yolov8n_hardhat_fp32.pt`
- Converted FP16 ONNX edge weights: `weights/yolov8n_hardhat_fp16.onnx`

Upload the model weights to Google Drive or Hugging Face and paste the public links here before final submission.

## Performance Benchmark Table

Benchmark collected locally on Apple M1 CPU using `imgsz=416`, `warmup=3`, and `runs=30`.

| Metric | FP32 YOLOv8n `.pt` | FP16 ONNX Edge Model | Change |
|---|---:|---:|---:|
| Model size (MB) | 23.29 | 5.83 | -17.46 |
| mAP50-95 | 0.3125 | 0.3037 | -0.0088 |
| FPS on local machine | 5.63 | 6.24 | +0.60 |

## Video Proof

- Local 30-second slow demo: `demo/hardhat_30sec_slow_inference.mp4`
- Final required upload link: `TODO: paste unlisted YouTube / Loom / Google Drive URL`

The final video should show `live_inference.py` running locally, point out the bounding boxes and the FPS/preprocess/inference/post-NMS overlay metrics, and explain why FP16 was chosen over INT8 for this edge conversion.
