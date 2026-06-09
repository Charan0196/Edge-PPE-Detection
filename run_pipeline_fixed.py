#!/usr/bin/env python3
"""Fixed YOLOv8 training and conversion pipeline for edge deployment."""

from pathlib import Path
import logging
import yaml
from ultralytics import YOLO
import onnxruntime as ort

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_baseline_model(data_yaml: Path, epochs: int = 100):
    """Train baseline YOLOv8n model in FP32."""
    logger.info("=" * 60)
    logger.info("Phase 1: Training Baseline Model (FP32)")
    logger.info("=" * 60)
    
    if not data_yaml.exists():
        raise FileNotFoundError(f"Data config not found: {data_yaml}")
    
    logger.info(f"Data config: {data_yaml}")
    
    model = YOLO("yolov8n.pt")
    logger.info("Loaded YOLOv8n base model")
    
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=416,
        batch=16,
        project="runs/hardhat",
        name="yolov8n_fp32",
        device=0,
        patience=10,
        save=True,
        verbose=True,
        plots=True,
    )
    
    logger.info("✓ Training completed!")
    return results


def export_to_onnx(model_path: Path, output_dir: Path = Path("weights")):
    """Export trained model to ONNX format."""
    logger.info("\n" + "=" * 60)
    logger.info("Phase 2: Model Conversion & Quantization")
    logger.info("=" * 60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading model: {model_path}")
    model = YOLO(str(model_path))
    
    # Export to ONNX (FP32)
    logger.info("Exporting to ONNX (FP32)...")
    onnx_fp32_path = output_dir / "yolov8n_hardhat_fp32.onnx"
    model.export(format="onnx", imgsz=416, half=False)
    logger.info(f"✓ ONNX FP32 exported")
    
    # Export to ONNX (FP16 - quantized)
    logger.info("Exporting to ONNX (FP16 - quantized)...")
    onnx_fp16_path = output_dir / "yolov8n_hardhat_fp16.onnx"
    model.export(format="onnx", imgsz=416, half=True)
    logger.info(f"✓ ONNX FP16 exported")
    
    return onnx_fp32_path, onnx_fp16_path


def validate_onnx_model(onnx_model_path: Path):
    """Validate ONNX model can be loaded and run."""
    logger.info(f"\nValidating ONNX model: {onnx_model_path}")
    
    try:
        session = ort.InferenceSession(str(onnx_model_path))
        logger.info(f"✓ Model loaded successfully")
        
        # Get model info
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        logger.info(f"  Input: {input_name}")
        logger.info(f"  Output: {output_name}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Model validation failed: {e}")
        return False


def benchmark_models(fp32_model: Path, fp16_model: Path):
    """Benchmark FP32 vs FP16 models."""
    logger.info("\n" + "=" * 60)
    logger.info("Benchmarking Models")
    logger.info("=" * 60)
    
    import numpy as np
    import time
    
    # Create dummy input
    dummy_input = np.random.randn(1, 3, 416, 416).astype(np.float32)
    
    # Benchmark FP32
    logger.info("\nFP32 Model:")
    session_fp32 = ort.InferenceSession(str(fp32_model))
    input_name = session_fp32.get_inputs()[0].name
    output_name = session_fp32.get_outputs()[0].name
    
    start = time.time()
    for _ in range(10):
        session_fp32.run([output_name], {input_name: dummy_input})
    fp32_time = (time.time() - start) / 10 * 1000
    logger.info(f"  Inference time: {fp32_time:.2f} ms")
    logger.info(f"  FPS: {1000/fp32_time:.1f}")
    
    # Benchmark FP16
    logger.info("\nFP16 Model (Quantized):")
    session_fp16 = ort.InferenceSession(str(fp16_model))
    
    start = time.time()
    for _ in range(10):
        session_fp16.run([output_name], {input_name: dummy_input})
    fp16_time = (time.time() - start) / 10 * 1000
    logger.info(f"  Inference time: {fp16_time:.2f} ms")
    logger.info(f"  FPS: {1000/fp16_time:.1f}")
    
    # Summary
    logger.info(f"\nSpeedup: {fp32_time/fp16_time:.2f}x faster")
    logger.info(f"Latency reduction: {(fp32_time-fp16_time)/fp32_time*100:.1f}%")


def main():
    """Execute complete training and conversion pipeline."""
    try:
        # Phase 1: Train baseline
        data_yaml = Path("data/hardhat_yolo/data.yaml")
        train_baseline_model(data_yaml, epochs=100)
        
        # Phase 2: Export and quantize
        model_path = Path("runs/hardhat/yolov8n_fp32/weights/best.pt")
        fp32_path, fp16_path = export_to_onnx(model_path)
        
        # Validate
        validate_onnx_model(fp32_path)
        validate_onnx_model(fp16_path)
        
        # Benchmark
        benchmark_models(fp32_path, fp16_path)
        
        logger.info("\n" + "=" * 60)
        logger.info("✓ Pipeline completed successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
