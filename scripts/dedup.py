"""
dedup.py

Removes near-duplicate frames from holedetector/frames/ (or any folder you point it at)
using perceptual hashing, and copies the kept frames into frames_dedup/.

Why: video extracted at 1fps often has long stretches of near-identical frames
(idle excavator, slow repositioning). Deduping before labeling saves a lot of
manual effort.

Usage:
    python dedup.py                          # defaults: frames/ -> frames_dedup/
    python dedup.py --input frames --output frames_dedup --threshold 5
"""

import argparse
import shutil
from pathlib import Path

from imagededup.methods import PHash

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "frames/leveling_slop"
DEFAULT_OUTPUT = PROJECT_ROOT / "frames_dedup/leveling_slop"


def collect_images(root: Path):
    exts = {".jpg", ".jpeg", ".png"}
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT),
                         help="Folder containing frames (searched recursively)")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT),
                         help="Folder to copy deduplicated frames into")
    parser.add_argument("--threshold", type=int, default=5,
                         help="Max hamming distance to count as a duplicate "
                              "(lower = stricter, keeps more frames; "
                              "higher = looser, keeps fewer). Default: 5")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    images = collect_images(input_dir)
    if not images:
        print(f"No images found under {input_dir}")
        return

    print(f"Found {len(images)} frames under {input_dir}")
    print("Hashing and finding duplicates (this can take a minute for thousands of frames)...")

    # imagededup wants a flat directory of images. If your frames are in
    # per-video subfolders, we hash each subfolder separately so frames from
    # different videos are never compared against each other (they'd never
    # be true duplicates anyway, and it keeps the run fast).
    subdirs = [d for d in input_dir.iterdir() if d.is_dir()] or [input_dir]

    phasher = PHash()
    output_dir.mkdir(parents=True, exist_ok=True)

    total_kept = 0
    total_seen = 0

    for subdir in subdirs:
        sub_images = collect_images(subdir) if subdir != input_dir else images
        if not sub_images:
            continue

        encodings = phasher.encode_images(image_dir=str(subdir))
        duplicates = phasher.find_duplicates(encoding_map=encodings, max_distance_threshold=args.threshold)

        kept = set()
        seen = set()
        for fname in sorted(duplicates.keys()):
            if fname in seen:
                continue
            kept.add(fname)
            seen.add(fname)
            for dup in duplicates[fname]:
                seen.add(dup)

        sub_out_dir = output_dir / subdir.name if subdir != input_dir else output_dir
        sub_out_dir.mkdir(parents=True, exist_ok=True)

        for fname in kept:
            src = subdir / fname
            if src.exists():
                shutil.copy2(src, sub_out_dir / fname)

        print(f"  {subdir.name}: {len(sub_images)} frames -> kept {len(kept)}")
        total_kept += len(kept)
        total_seen += len(sub_images)

    print(f"\nDone. Kept {total_kept} of {total_seen} frames "
          f"({total_kept/total_seen:.0%}). Output: {output_dir}")
    print("If too many/few were kept, re-run with a different --threshold "
          "(lower = keeps more, higher = keeps fewer).")


if __name__ == "__main__":
    main()