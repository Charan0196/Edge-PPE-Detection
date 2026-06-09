# Live Inference Documentation

## Overview
The `live_inference.py` script performs real-time object detection on video feeds or webcam input using an edge-optimized ONNX model. It displays bounding boxes, confidence scores, and performance metrics directly on the video output.

## Features
- **Real-time Inference**: Process video frames at 50+ FPS on CPU
- **ONNX Runtime Support**: Compatible with quantized edge models
- **Performance Metrics**: Display FPS, preprocessing, inference, and post-processing latency
- **Flexible Input**: Support for video files, webcam, or IP camera streams
- **NMS Filtering**: Non-Maximum Suppression for overlapping detections
- **Confidence & IoU Thresholds**: Configurable detection parameters

## Prerequisites
```bash
pip install -r requirements.txt
```

Required packages:
- opencv-python
- numpy
- onnxruntime
- pyyaml

## Usage

### Basic Usage (Video File)
```bash
python live_inference.py \
  --model weights/yolov8n_hardhat_fp16.onnx \
  --source data/demo/hardhat_test_video.mp4 \
  --data data/hardhat_yolo/data.yaml
```

### Webcam Inference
```bash
python live_inference.py \
  --model weights/yolov8n_hardhat_fp16.onnx \
  --source 0 \
  --data data/hardhat_yolo/data.yaml
```

### Advanced Options
```bash
python live_inference.py \
  --model weights/yolov8n_hardhat_fp16.onnx \
  --source data/demo/hardhat_test_video.mp4 \
  --data data/hardhat_yolo/data.yaml \
  --imgsz 416 \
  --conf 0.25 \
  --iou 0.45 \
  --providers CPUExecutionProvider
```

## Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | Path | Required | Path to ONNX model file |
| `--source` | str | "0" | Video file path or webcam index (0) |
| `--data` | Path | None | Path to YOLO data.yaml with class names |
| `--imgsz` | int | 416 | Input image size (must match model) |
| `--conf` | float | 0.25 | Confidence threshold (0-1) |
| `--iou` | float | 0.45 | IoU threshold for NMS (0-1) |
| `--providers` | list | Auto | ONNX Runtime providers |

## Supported ONNX Runtime Providers

| Provider | Hardware | Note |
|----------|----------|------|
| `CPUExecutionProvider` | CPU | Universal, works everywhere |
| `CUDAExecutionProvider` | NVIDIA GPU | Fast GPU inference |
| `TensorrtExecutionProvider` | NVIDIA GPU | Optimized TensorRT |
| `CoreMLExecutionProvider` | Apple Neural Engine | macOS/iOS |

### Example: GPU Inference
```bash
python live_inference.py \
  --model weights/yolov8n_hardhat_fp16.onnx \
  --source 0 \
  --providers CUDAExecutionProvider
```

## Real-time Metrics Displayed

The script overlays the following metrics on each frame:

### 1. Bounding Boxes
- Colored rectangles around detected objects
- Class label with confidence score
- Color-coded by class (green=helmet, orange=no-helmet, etc.)

### 2. Performance Metrics (top-left corner)
```
Inference FPS: 58.2
Preprocess: 8.20 ms
Inference: 17.22 ms
Post/NMS: 3.08 ms
```

**Metrics Explanation**:
- **Inference FPS**: Frames per second (excluding rendering/display time)
- **Preprocess**: Time to resize, pad, and normalize input
- **Inference**: ONNX model inference time
- **Post/NMS**: Bounding box postprocessing and NMS filtering

## Pipeline Overview

```
Input Frame (640x480)
        ↓
[Preprocess] - Letterbox resize + normalize
        ↓
[Inference] - ONNX model prediction
        ↓
[Postprocess] - Convert YOLO format + threshold filtering
        ↓
[NMS] - Non-Maximum Suppression
        ↓
[Visualization] - Draw boxes + metrics overlay
        ↓
Output Frame with Annotations
```

## Implementation Details

### Preprocessing (Letterbox)
- Maintains aspect ratio while resizing to model input size
- Pads with gray border (114, 114, 114)
- Converts BGR → RGB
- Normalizes to [0, 1] range
- Transposes to CHW format for model

### Inference
- Single-batch inference through ONNX Runtime
- Output: [1, N, 85] tensor where N=predictions
- Classes: detection with (x, y, w, h, confidence, class_scores)

### Postprocessing
- Converts YOLO format (cx, cy, w, h) → XYXY format
- Filters by confidence threshold
- Scales coordinates back to original frame size
- Applies Non-Maximum Suppression (NMS)

### Performance Optimization
- Avoided rendering time in FPS calculation
- Uses OpenCV's optimized NMS
- Minimal memory allocation per frame
- CPU-friendly operations

## Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit inference |
| `Esc` | Quit inference |

## Troubleshooting

### Model Not Found
```
RuntimeError: Could not load ONNX model
```
**Solution**: Verify model path and ONNX file exists
```bash
ls weights/yolov8n_hardhat_fp16.onnx
```

### Video Source Error
```
RuntimeError: Could not open video source
```
**Solution**: Check file path or webcam availability
```bash
# For webcam: verify camera is connected
# For file: verify file exists and format is supported
```

### ONNX Runtime Error
```
RuntimeError: No provider found
```
**Solution**: Install required provider
```bash
pip install onnxruntime onnxruntime-gpu  # For GPU
```

### Low FPS Performance
- Try CPU provider (often faster than GPU for small models)
- Reduce confidence threshold (processes fewer detections)
- Use smaller input size (--imgsz 320 instead of 416)

## Performance Benchmarks

### FP32 Model
- Model Size: 24.5 MB
- Inference Time: 22.2 ms/frame
- FPS: ~45 FPS (CPU)
- Memory Usage: ~850 MB

### FP16 Model (Quantized)
- Model Size: 12.3 MB (50% reduction)
- Inference Time: 17.2 ms/frame
- FPS: ~58 FPS (CPU)
- Memory Usage: ~480 MB (43% reduction)

**Improvement**:
- 28.9% faster inference
- 50% smaller model
- 43.5% less memory
- 0.77% accuracy loss (trade-off)

## Best Practices

1. **Match Model Input Size**: Always use `--imgsz` matching model training size (416)
2. **CPU vs GPU**: Test both for your specific model and hardware
3. **Confidence Threshold**: Start with 0.25, adjust based on use case
4. **IoU Threshold**: Default 0.45 works well, lower for dense detections
5. **Monitor Metrics**: Watch FPS and latency to identify bottlenecks

## Advanced Usage

### Analyzing Metrics
Export metrics to CSV for analysis:
```python
# Modify live_inference.py to log metrics
with open('metrics.csv', 'a') as f:
    f.write(f"{fps},{preprocess_ms},{inference_ms},{postprocess_ms}\n")
```

### Custom Class Colors
Modify `colors` list in `draw_overlay()`:
```python
colors = [(B,G,R), (B,G,R), ...]  # BGR format
```

### Processing Only Every N Frames
```python
if frame_idx % 2 == 0:  # Process every 2nd frame
    detections = postprocess(...)
```

## Limitations

- Single-threaded inference (can be parallelized)
- No tracking across frames
- No context/temporal information
- Fixed input size (no dynamic batching)
- No multi-model ensemble

## Future Improvements

- [ ] Multi-threading for preprocessing
- [ ] Object tracking across frames
- [ ] Batch inference for multiple videos
- [ ] Export detections to JSON/CSV
- [ ] Web dashboard for metrics visualization
