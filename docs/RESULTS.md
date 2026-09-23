# Training Results

## Run history

| Run | Labeled images | Mask precision | Mask recall | mAP50 (M) | mAP50-95 (M) |
|---|---|---|---|---|---|
| `hole_detector` (Aug 26) | 43 (37 train / 6 val) | 0.82 | 0.23 | 0.24 | 0.14 |
| `hole_detector-3` (Sep 21) | 142 | 0.77 | 0.32 | 0.34 | 0.22 |

Model: YOLOv8n-seg, 100 epochs, single class (`slop`), trained on an RTX 3060 Ti.

## Plots (`docs/results/`)

- **`results.png`** — full training curves (box/seg/cls/dfl loss, precision, recall, mAP50, mAP50-95) for both box and mask heads across all 100 epochs.
- **`MaskPR_curve.png`** — mask precision/recall tradeoff across confidence thresholds.
- **`confusion_matrix_normalized.png`** — normalized confusion matrix for the `slop` class vs. background.

## Reading the latest run

Training converged cleanly — all losses flatten by ~epoch 30 on both train and val, with no overfitting blowup. But precision and recall plateau at the same point: **precision ~0.77-0.82** (the model is confidently right when it fires) against **recall only ~0.32** (it still misses roughly two-thirds of actual holes). Since the curves are flat well before epoch 100, more epochs on this dataset will not move these numbers further.

## What can be improved

1. **More labeled data is the primary lever.** Only 142 of the 1,900 deduplicated `Leveling_Slop` frames have been labeled via `label_sam2.py`. The recall ceiling tracks dataset size directly (0.23 → 0.32 going from 43 → 142 images), so continuing to label more of the remaining ~1,750 frames is the highest-value next step.
2. **`Storm_Drainage.mp4` is still untouched.** Only `Leveling_Slop` has been extracted/deduped/labeled. Bringing in a second video adds scene/lighting variety the model hasn't seen, which should generalize recall better than more frames from the same footage alone.
3. **Class imbalance / hard negatives.** Background points during SAM2 labeling (right-click) help the model learn what isn't a hole — worth being deliberate about clicking clear negatives (spoil piles, shadows, flat dirt) on frames that don't contain a hole, rather than only skipping them.
4. **Confidence threshold tuning.** Given the precision/recall tradeoff, running `detect.py` with a lower `--conf` (e.g. 0.15 instead of the 0.25 default) trades some false positives for better recall as a stopgap while more data is collected — already validated informally during video testing.
5. **Consider a larger base model** (`yolov8s-seg` or `yolov8m-seg`) once the dataset is bigger — `yolov8n-seg` is the smallest/fastest variant and may itself be capacity-limited for this task, but it's not worth switching before addressing the data volume issue first.
