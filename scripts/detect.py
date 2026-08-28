"""
detect.py

Run your trained hole-detector model on images or a video and see the results.

Usage:
    # single image
    python detect.py --weights runs/hole_detector/weights/best.pt --source frames_dedup/video1/frame_00010.jpg

    # folder of images
    python detect.py --weights runs/hole_detector/weights/best.pt --source frames_dedup/video1

    # video file (draws masks frame by frame, saves an annotated output video)
    python detect.py --weights runs/hole_detector/weights/best.pt --source videos/my_clip.mp4

    # webcam / live feed
    python detect.py --weights runs/hole_detector/weights/best.pt --source 0
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "hole_detector" / "weights" / "best.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS),
                         help="Path to trained model weights (best.pt)")
    parser.add_argument("--source", type=str, required=True,
                         help="Image, folder, video file, or '0' for webcam")
    parser.add_argument("--conf", type=float, default=0.25,
                         help="Confidence threshold (default 0.25). Raise it if you're "
                              "getting false positives, lower it if it's missing real holes.")
    parser.add_argument("--save-dir", type=str, default=str(PROJECT_ROOT / "runs" / "detect"),
                         help="Where annotated output gets saved")
    parser.add_argument("--show", action="store_true",
                         help="Pop up a live window while processing (useful for video/webcam)")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"Weights not found at {weights_path}. "
              f"Check the path or pass --weights explicitly.")
        return

    print(f"Loading model from {weights_path} ...")
    model = YOLO(str(weights_path))

    print(f"Running inference on: {args.source}")
    results = model.predict(
        source=args.source,
        conf=args.conf,
        save=True,          # saves annotated images/video to disk
        project=args.save_dir,
        name="run",
        show=args.show,
        stream=True,        # memory-efficient for videos/large folders
    )

    total_detections = 0
    total_frames = 0
    for result in results:
        total_frames += 1
        n = len(result.boxes) if result.boxes is not None else 0
        total_detections += n

    print(f"\nProcessed {total_frames} frame(s), {total_detections} hole detection(s) total.")
    print(f"Annotated output saved under: {args.save_dir}/run")


if __name__ == "__main__":
    main()
