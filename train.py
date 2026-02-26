"""
YOLOv13 Training Script

Usage:
    python train.py                                          # Train with default config
    python train.py --model yolov13_sar.yaml --data rsar.yaml
    python train.py --model yolov13.yaml --data coco.yaml --epochs 300
    python train.py --weights yolov13n.pt --data coco.yaml   # Fine-tune from pretrained

Examples:
    # Train YOLOv13-SAR on RSAR dataset
    python train.py --model ultralytics/cfg/models/v13/yolov13_sar.yaml \
                    --data ultralytics/cfg/datasets/rsar.yaml \
                    --imgsz 800 --batch 16 --epochs 200

    # Resume training from a checkpoint
    python train.py --resume runs/train/exp/weights/last.pt
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv13 Training")

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="ultralytics/cfg/models/v13/yolov13_sar.yaml",
        help="Path to model config (.yaml) or pretrained weights (.pt)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="obb",
        choices=["detect", "segment", "classify", "pose", "obb"],
        help="YOLO task type (default: obb)",
    )

    # Dataset
    parser.add_argument(
        "--data",
        type=str,
        default="ultralytics/cfg/datasets/rsar.yaml",
        help="Path to dataset config (.yaml)",
    )

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=800, help="Input image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (-1 for auto)")
    parser.add_argument("--device", type=str, default="0", help="CUDA device(s), e.g. '0' or '0,1,2,3' or 'cpu'")
    parser.add_argument("--workers", type=int, default=8, help="Number of dataloader workers")

    # Optimizer
    parser.add_argument("--optimizer", type=str, default="SGD", help="Optimizer: SGD, Adam, AdamW, etc.")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final learning rate factor (lr0 * lrf)")
    parser.add_argument("--warmup-epochs", type=float, default=3.0, help="Warmup epochs")

    # Augmentation
    parser.add_argument("--degrees", type=float, default=180.0, help="Rotation augmentation degrees")
    parser.add_argument("--flipud", type=float, default=0.5, help="Vertical flip probability")
    parser.add_argument("--fliplr", type=float, default=0.5, help="Horizontal flip probability")
    parser.add_argument("--mosaic", type=float, default=1.0, help="Mosaic augmentation probability")
    parser.add_argument("--scale", type=float, default=0.5, help="Scale augmentation factor")
    parser.add_argument("--mixup", type=float, default=0.15, help="Mixup augmentation probability")
    parser.add_argument("--copy-paste", type=float, default=0.5, help="Copy-paste augmentation probability")

    # Output
    parser.add_argument("--project", type=str, default="runs/train", help="Output project directory")
    parser.add_argument("--name", type=str, default="exp", help="Experiment name")

    # Training control
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience (0 to disable)")
    parser.add_argument("--save-period", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--resume", type=str, default="", help="Resume training from checkpoint (.pt)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Resume from checkpoint if specified
    if args.resume:
        model = YOLO(args.resume)
        model.train(resume=True)
        return

    # Build model from config
    model = YOLO(args.model, task=args.task)

    # Start training
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        # Optimizer
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        warmup_epochs=args.warmup_epochs,
        # Augmentation
        degrees=args.degrees,
        flipud=args.flipud,
        fliplr=args.fliplr,
        mosaic=args.mosaic,
        scale=args.scale,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        # Output
        project=args.project,
        name=args.name,
        # Training control
        patience=args.patience,
        save_period=args.save_period,
        val=True,
        plots=True,
    )


if __name__ == "__main__":
    main()
