#!/usr/bin/env python3
"""Benchmark YOLOv8 models (FP32 vs quantized)."""

import argparse
import time
from pathlib import Path
import numpy as np
import onnxruntime as ort


def benchmark_onnx_model(model_path: Path, num_runs: int = 100, imgsz: int = 416):
    """Benchmark ONNX model inference latency and throughput."""
    
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_path.name}")
    print(f"{'='*60}")
    
    # Load model
    session = ort.InferenceSession(str(model_path))
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    print(f"Model: {model_path}")
    print(f"Input shape: {session.get_inputs()[0].shape}")
    print(f"Output shape: {session.get_outputs()[0].shape}")
    
    # Prepare input
    dummy_input = np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        session.run([output_name], {input_name: dummy_input})
    
    # Benchmark
    latencies = []
    start = time.perf_counter()
    for _ in range(num_runs):
        t0 = time.perf_counter()
        session.run([output_name], {input_name: dummy_input})
        latencies.append((time.perf_counter() - t0) * 1000)
    total_time = time.perf_counter() - start
    
    # Statistics
    latencies = np.array(latencies)
    print(f"\nResults ({num_runs} runs):")
    print(f"  Min: {latencies.min():.2f} ms")
    print(f"  Max: {latencies.max():.2f} ms")
    print(f"  Mean: {latencies.mean():.2f} ms")
    print(f"  Std: {latencies.std():.2f} ms")
    print(f"  Median: {np.median(latencies):.2f} ms")
    print(f"  FPS: {1000/latencies.mean():.1f}")
    
    return latencies


def main():
    parser = argparse.ArgumentParser(description="Benchmark ONNX models")
    parser.add_argument("--model", type=Path, help="Model path")
    parser.add_argument("--runs", type=int, default=100, help="Number of inference runs")
    parser.add_argument("--imgsz", type=int, default=416, help="Input image size")
    parser.add_argument("--compare", action="store_true", help="Compare FP32 vs FP16")
    
    args = parser.parse_args()
    
    if args.compare:
        # Compare two models
        fp32_model = Path("weights/yolov8n_hardhat_fp32.onnx")
        fp16_model = Path("weights/yolov8n_hardhat_fp16.onnx")
        
        if not fp32_model.exists() or not fp16_model.exists():
            print(f"Error: Model files not found")
            return
        
        fp32_latencies = benchmark_onnx_model(fp32_model, args.runs, args.imgsz)
        fp16_latencies = benchmark_onnx_model(fp16_model, args.runs, args.imgsz)
        
        print(f"\n{'='*60}")
        print("Comparison Summary")
        print(f"{'='*60}")
        print(f"FP32 Mean: {fp32_latencies.mean():.2f} ms ({1000/fp32_latencies.mean():.1f} FPS)")
        print(f"FP16 Mean: {fp16_latencies.mean():.2f} ms ({1000/fp16_latencies.mean():.1f} FPS)")
        print(f"Speedup: {fp32_latencies.mean()/fp16_latencies.mean():.2f}x")
        print(f"Latency reduction: {(fp32_latencies.mean()-fp16_latencies.mean())/fp32_latencies.mean()*100:.1f}%")
        
    else:
        if not args.model:
            print("Error: --model required or use --compare")
            return
        benchmark_onnx_model(args.model, args.runs, args.imgsz)


if __name__ == "__main__":
    main()
