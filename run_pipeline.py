#!/usr/bin/env python3
"""YOLOv8 training pipeline for hard hat detection."""

from pathlib import Path
import yaml
from ultralytics import YOLO


def main():
    """Train YOLOv8 model on hard hat dataset."""
    
    # Configuration
    model_name = "yolov8n"
    data_yaml = Path("data/hardhat_yolo/data.yaml")
    project_dir = Path("runs/hardhat")
    epochs = 100
    imgsz = 416
    batch_size = 16
    device = 0  # GPU device, use 'cpu' for CPU
    
    print("=" * 60)
    print("YOLOv8 Hard Hat Detection Training Pipeline")
    print("=" * 60)
    
    # Verify data.yaml exists
    if not data_yaml.exists():
        raise FileNotFoundError(f"Data config not found: {data_yaml}")
    
    print(f"\n✓ Data config found: {data_yaml}")
    
    # Load base model
    print(f"\n→ Loading {model_name} model...")
    model = YOLO(f"{model_name}.pt")
    
    # Train model
    print(f"\n→ Starting training...")
    print(f"  Epochs: {epochs}")
    print(f"  Image size: {imgsz}")
    print(f"  Batch size: {batch_size}")
    
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        project=str(project_dir),
        name="yolov8n_fp32",
        device=device,
        patience=10,
        save=True,
        verbose=True,
        plots=True,
    )
    
    print("\n✓ Training completed!")
    print(f"✓ Model saved to: {project_dir}")
    
    # Validate model
    print("\n→ Running validation...")
    val_results = model.val()
    print("✓ Validation completed!")
    
    return results


if __name__ == "__main__":
    main()
