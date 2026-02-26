"""
YOLOv13 Validation / Testing Script

Usage:
    python val.py --weights runs/train/exp/weights/best.pt
    python val.py --weights yolov13_sar.pt --data rsar.yaml --split test
    python val.py --weights yolov13_sar.pt --iou 0.5 --conf 0.001

Examples:
    # Evaluate on test split with default settings
    python val.py --weights runs/train/exp/weights/best.pt \
                  --data ultralytics/cfg/datasets/rsar.yaml \
                  --split test --imgsz 800

    # Evaluate with custom IoU threshold
    python val.py --weights runs/train/exp/weights/best.pt \
                  --iou 0.5 --conf 0.001 --save-json
"""

import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv13 Validation / Testing")

    # Model
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to model weights (.pt)",
    )

    # Dataset
    parser.add_argument(
        "--data",
        type=str,
        default="ultralytics/cfg/datasets/rsar.yaml",
        help="Path to dataset config (.yaml)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Dataset split to evaluate on (default: test)",
    )

    # Evaluation settings
    parser.add_argument("--imgsz", type=int, default=800, help="Input image size")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--device", type=str, default="0", help="CUDA device, e.g. '0' or 'cpu'")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for mAP calculation")
    parser.add_argument("--conf", type=float, default=0.05, help="Confidence threshold")

    # Output
    parser.add_argument("--project", type=str, default="runs/val", help="Output project directory")
    parser.add_argument("--name", type=str, default="exp", help="Experiment name")
    parser.add_argument("--save-json", action="store_true", help="Save results in COCO JSON format")
    parser.add_argument("--plots", action="store_true", default=True, help="Generate evaluation plots")
    parser.add_argument("--verbose", action="store_true", default=True, help="Print detailed per-class metrics")

    return parser.parse_args()


def main():
    args = parse_args()

    # Load model
    model = YOLO(args.weights)

    # Run evaluation
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        iou=args.iou,
        conf=args.conf,
        project=args.project,
        name=args.name,
        save_json=args.save_json,
        plots=args.plots,
        verbose=args.verbose,
    )

    # Print summary
    print("\n" + "=" * 50)
    print("Evaluation Results:")
    print(f"  mAP50:       {metrics.box.map50:.4f}")
    print(f"  mAP50-95:    {metrics.box.map:.4f}")
    print(f"  Precision:   {metrics.box.mp:.4f}")
    print(f"  Recall:      {metrics.box.mr:.4f}")
    if hasattr(metrics.box, "f1") and len(metrics.box.f1) > 0:
        print(f"  Mean F1:     {metrics.box.f1.mean():.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
