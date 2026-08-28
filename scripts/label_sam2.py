"""
label_sam2.py
 
Interactive labeling tool using SAM2's image predictor.
 
For each frame in frames_dedup/:
  - Shows the image
  - You click on the dirt slop (left click = foreground point, right click = background point)
  - Press ENTER to accept the mask and save it
  - Press 'n' to skip this frame (no dirt slop / not usable)
  - Press 'r' to reset points on the current frame and try again
  - Press 'q' to quit and save progress
 
Masks are saved as binary PNGs in masks/, matching the frame filename.
A running log (labeled.txt) tracks which frames have been processed so you
can stop and resume later without re-labeling frames.
 
Requirements:
    pip install git+https://github.com/facebookresearch/sam2.git
    Download a SAM2 checkpoint (see SAM2 repo) and set CHECKPOINT_PATH / MODEL_CONFIG below.
 
Usage:
    python label_sam2.py
    python label_sam2.py --input frames_dedup --output masks
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "frames_dedup/leveling_slop"
DEFAULT_OUTPUT = PROJECT_ROOT / "masks"

# --- Update these to match where you downloaded the SAM2 checkpoint/config ---
# Get checkpoints from: https://github.com/facebookresearch/sam2#model-description
CHECKPOINT_PATH = str(PROJECT_ROOT / "sam2.1_hiera_large.pt")
MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
# -------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
 
 
class ClickCollector:
    def __init__(self):
        self.points = []
        self.labels = []  # 1 = foreground, 0 = background
 
    def callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append([x, y])
            self.labels.append(1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.points.append([x, y])
            self.labels.append(0)
 
    def reset(self):
        self.points = []
        self.labels = []
 
    def as_arrays(self):
        return np.array(self.points), np.array(self.labels)
 
 
def overlay_mask(image, mask, alpha=0.5, color=(0, 255, 0)):
    overlay = image.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
 
 
def draw_points(image, points, labels):
    out = image.copy()
    for (x, y), lbl in zip(points, labels):
        color = (0, 255, 0) if lbl == 1 else (0, 0, 255)
        cv2.circle(out, (int(x), int(y)), 5, color, -1)
    return out
 
 
def load_progress(log_path: Path):
    if not log_path.exists():
        return set()
    with open(log_path, "r") as f:
        return set(line.strip() for line in f if line.strip())
 
 
def append_progress(log_path: Path, filename: str):
    with open(log_path, "a") as f:
        f.write(filename + "\n")
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
 
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "labeled.txt"
 
    exts = {".jpg", ".jpeg", ".png"}
    frames = sorted([p for p in input_dir.rglob("*") if p.suffix.lower() in exts])
    if not frames:
        print(f"No frames found in {input_dir}")
        return
 
    done = load_progress(log_path)
    remaining = [f for f in frames if f.name not in done]
    print(f"{len(frames)} total frames, {len(done)} already labeled, {len(remaining)} remaining.")
 
    if not remaining:
        print("Nothing left to label.")
        return
 
    print(f"Loading SAM2 on {DEVICE} (this can take a moment)...")
    sam2_model = build_sam2(MODEL_CONFIG, CHECKPOINT_PATH, device=DEVICE)
    predictor = SAM2ImagePredictor(sam2_model)
 
    clicker = ClickCollector()
    window_name = "SAM2 Slop Labeler  |  L-click=slop  R-click=background  ENTER=save  n=skip  r=reset  q=quit"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, clicker.callback)
 
    for frame_path in remaining:
        image_bgr = cv2.imread(str(frame_path))
        if image_bgr is None:
            print(f"  Could not read {frame_path.name}, skipping.")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)
 
        clicker.reset()
        current_mask = None
 
        while True:
            display = image_bgr.copy()
            if current_mask is not None:
                display = overlay_mask(display, current_mask)
            display = draw_points(display, clicker.points, clicker.labels)
            cv2.putText(display, frame_path.name, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow(window_name, display)
 
            key = cv2.waitKey(30) & 0xFF
 
            if key == ord("q"):
                print("Quitting, progress saved.")
                cv2.destroyAllWindows()
                return
 
            elif key == ord("n"):
                print(f"  Skipped {frame_path.name}")
                append_progress(log_path, frame_path.name)
                break
 
            elif key == ord("r"):
                clicker.reset()
                current_mask = None
 
            elif key == 13:  # ENTER
                if current_mask is None:
                    print("  No mask to save yet - click on the slop first.")
                    continue
                out_path = output_dir / (frame_path.stem + "_mask.png")
                cv2.imwrite(str(out_path), (current_mask * 255).astype(np.uint8))
                append_progress(log_path, frame_path.name)
                print(f"  Saved mask for {frame_path.name}")
                break
 
            # Re-run prediction whenever the point set changes
            if clicker.points:
                pts, lbls = clicker.as_arrays()
                masks, scores, _ = predictor.predict(
                    point_coords=pts,
                    point_labels=lbls,
                    multimask_output=True,
                )
                # keep the highest-scoring mask
                best = np.argmax(scores)
                current_mask = masks[best]
 
    print("\nAll frames processed.")
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()
 

