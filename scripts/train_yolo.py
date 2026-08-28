"""
train_yolo.py

Fine-tunes a pretrained YOLO segmentation model on your slop dataset.

Usage:
    python train_yolo.py
    python train_yolo.py --model yolov8s-seg.pt --epochs 100 --imgsz 640
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_YAML = PROJECT_ROOT / "dataset" / "data.yaml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_YAML))
    parser.add_argument("--model", type=str, default="yolov8n-seg.pt",
                         help="Pretrained checkpoint to fine-tune from. "
                              "n=nano (fastest, least accurate) up to x=extra-large. "
                              "Good default for a small first dataset: yolov8n-seg.pt or yolov8s-seg.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", type=str, default=str(PROJECT_ROOT / "runs"))
    parser.add_argument("--name", type=str, default="hole_detector")
    args = parser.parse_args()

    if not Path(args.data).exists():
        print(f"data.yaml not found at {args.data}. Run prepare_yolo_dataset.py first.")
        return

    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        patience=20,  # stop early if val performance plateaus
    )

    print(f"\nTraining complete. Best weights saved under: "
          f"{args.project}/{args.name}/weights/best.pt")
    print("Use best.pt to run inference on new frames, or as the starting point "
          "for pre-labeling the rest of your dataset.")


if __name__ == "__main__":
    main()
