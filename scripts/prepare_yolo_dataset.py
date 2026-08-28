"""
prepare_yolo_dataset.py

Converts binary mask PNGs (from label_sam2.py) into YOLO-segmentation format
polygon labels, matches them back up with their source images, and builds a
train/val split ready for Ultralytics YOLO training.

Expected input:
    frames_dedup/<video_name>/frame_00001.jpg   (source images)
    masks/frame_00001_mask.png                  (binary masks, 255 = slop, 0 = background)

Output:
    dataset/
        images/train/*.jpg
        images/val/*.jpg
        labels/train/*.txt   (YOLO-seg polygon format)
        labels/val/*.txt
        data.yaml

Usage:
    python prepare_yolo_dataset.py
    python prepare_yolo_dataset.py --val-split 0.15 --min-area 50
"""

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FRAMES_DIR = PROJECT_ROOT / "frames_dedup"
DEFAULT_MASKS_DIR = PROJECT_ROOT / "masks"
DEFAULT_DATASET_DIR = PROJECT_ROOT / "dataset"

CLASS_NAMES = ["slop"]  # single class for now; extend later if needed (e.g. "spoil_pile")


def find_source_image(frame_stem: str, frames_dir: Path):
    """Masks are named <frame_stem>_mask.png; the source image with the same
    stem could be in any per-video subfolder under frames_dir, or directly
    inside frames_dir if it isn't split into subfolders."""
    exts = [".jpg", ".jpeg", ".png"]
    for ext in exts:
        direct = frames_dir / f"{frame_stem}{ext}"
        if direct.exists():
            return direct
    for candidate in frames_dir.rglob(f"{frame_stem}.*"):
        if candidate.suffix.lower() in exts:
            return candidate
    return None


def mask_to_yolo_polygons(mask: np.ndarray, img_w: int, img_h: int, min_area: float):
    """Extract external contours from a binary mask and normalize to YOLO-seg format:
    class_id x1 y1 x2 y2 ... (all coords normalized 0-1)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue  # skip tiny noise blobs from imperfect masks
        # Simplify polygon slightly to avoid huge point counts from noisy mask edges
        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3:
            continue
        coords = []
        for point in approx.reshape(-1, 2):
            x_norm = point[0] / img_w
            y_norm = point[1] / img_h
            coords.extend([f"{x_norm:.6f}", f"{y_norm:.6f}"])
        lines.append("0 " + " ".join(coords))  # class_id 0 = "hole"
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=str, default=str(DEFAULT_FRAMES_DIR))
    parser.add_argument("--masks", type=str, default=str(DEFAULT_MASKS_DIR))
    parser.add_argument("--output", type=str, default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--val-split", type=float, default=0.15,
                         help="Fraction of labeled frames held out for validation (default: 0.15)")
    parser.add_argument("--min-area", type=float, default=50,
                         help="Minimum contour area in pixels to keep as a valid polygon "
                              "(filters out tiny noise specks in the mask). Default: 50")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frames_dir = Path(args.frames)
    masks_dir = Path(args.masks)
    output_dir = Path(args.output)

    mask_files = sorted(masks_dir.glob("*_mask.png"))
    if not mask_files:
        print(f"No mask files found in {masks_dir} (expected *_mask.png). "
              f"Run label_sam2.py first.")
        return

    print(f"Found {len(mask_files)} labeled masks.")

    pairs = []  # (image_path, label_lines)
    skipped_no_image = 0
    skipped_no_polygons = 0

    for mask_path in mask_files:
        frame_stem = mask_path.stem.replace("_mask", "")
        image_path = find_source_image(frame_stem, frames_dir)
        if image_path is None:
            skipped_no_image += 1
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        # mask may not be exactly the same resolution as the source image if
        # anything got resized along the way - resize mask to match to be safe
        if mask.shape[:2] != (img_h, img_w):
            mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

        binary_mask = (mask > 127).astype(np.uint8) * 255
        lines = mask_to_yolo_polygons(binary_mask, img_w, img_h, args.min_area)

        if not lines:
            skipped_no_polygons += 1
            continue

        pairs.append((image_path, lines))

    print(f"Matched {len(pairs)} image/label pairs.")
    if skipped_no_image:
        print(f"  Skipped {skipped_no_image} masks with no matching source image found.")
    if skipped_no_polygons:
        print(f"  Skipped {skipped_no_polygons} masks with no polygons above min-area "
              f"(likely empty/noise-only masks).")

    if not pairs:
        print("Nothing to build a dataset from. Check that frames_dedup/ still contains "
              "the original images and paths match.")
        return

    random.seed(args.seed)
    random.shuffle(pairs)
    val_count = max(1, int(len(pairs) * args.val_split))
    val_pairs = pairs[:val_count]
    train_pairs = pairs[val_count:]

    for split_name, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
        (output_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)

        for image_path, lines in split_pairs:
            # prefix with parent folder name to avoid collisions between
            # frame_00001.jpg from different source videos
            unique_name = f"{image_path.parent.name}_{image_path.name}"
            unique_name = unique_name.replace(" ", "_")
            stem = Path(unique_name).stem

            dest_img = output_dir / "images" / split_name / unique_name
            shutil.copy2(image_path, dest_img)

            dest_label = output_dir / "labels" / split_name / f"{stem}.txt"
            with open(dest_label, "w") as f:
                f.write("\n".join(lines) + "\n")

    # data.yaml for Ultralytics
    data_yaml = output_dir / "data.yaml"
    with open(data_yaml, "w") as f:
        f.write(f"path: {output_dir.resolve()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(CLASS_NAMES)}\n")
        f.write(f"names: {CLASS_NAMES}\n")

    print(f"\nDataset built: {len(train_pairs)} train / {len(val_pairs)} val images.")
    print(f"Output: {output_dir}")
    print(f"data.yaml written to: {data_yaml}")


if __name__ == "__main__":
    main()
