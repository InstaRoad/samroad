# Training SAM-Road on the Kaggle "Sentinel-2 Roads" dataset

SAM-Road needs, per tile: an RGB image, a road mask, a keypoint mask, and a road
**graph** (it predicts a keypoint heatmap + topology, not just segmentation). The
Kaggle dataset [`sonisuyash/sentinel-2-roads-dataset`](https://www.kaggle.com/datasets/sonisuyash/sentinel-2-roads-dataset)
ships only raster masks (no road vectors), so `prepare_s2roads.py` **recovers the
graph from each binary mask** by skeletonising it and tracing the skeleton, then
writes the PNG + pickle layout SAM-Road's loader understands.

Every 256×256 tile becomes one SAM-Road tile (`IMAGE_SIZE = PATCH_SIZE = 256`,
margin 0 — the sampler uses the whole tile), exactly like the `s2rosa` branch.

## Dataset layout (raw)

The download is wrapped in a numeric parent folder and every leaf folder is doubled:

```
<root>/15765738/images_png/images_png/<id>.png            256×256 8-bit RGB (true colour)
       15765738/images_enhanced_png/images_enhanced_png/<id>.png  contrast-stretched RGB
       15765738/masks_png/masks_png/<id>.png              256×256 0/255 binary road mask
       15765738/images_tif/...  15765738/masks_tif/...     georeferenced float32 / uint8 (unused)
```

`prepare_s2roads.py` finds the leaf folders automatically (glob), so you pass the
download root and it copes with the wrapper + doubling.

## What the converter does

- **Image**: copies the chosen RGB variant verbatim (`--images images_png` or
  `images_enhanced_png`). Already 256×256 8-bit, no re-encode.
- **Road mask** (seg target): `masks_png` **dilated by `--road_buffer` px** (default 2,
  ≈ 5 px / 50 m band). The raw masks are ~1 px wide — too thin a segmentation target
  (severe class imbalance, thin-structure loss). The buffer only widens the training
  target; the **graph and keypoints are still traced from the original thin mask** so
  the centreline stays accurate. `--road_buffer 0` keeps the verbatim thin mask.
- **Graph**: `skeletonize` (scikit-image) → trace to a node/edge graph (`sknw`) →
  prune short dead-end spurs (skeleton noise) → chain each simplified edge polyline
  into consecutive integer-pixel edges. Result is the sat2graph adjacency dict
  `{(row,col): [(row,col), …]}` (SAM-Road's `coord_transform v[:, ::-1]` makes them x,y).
- **Keypoints**: graph nodes with degree ≠ 2 (intersections + dead-ends), excluding
  the patch border (roads cut by the window edge are not real keypoints).
- **Split**: the dataset has no published split, so a deterministic random
  80/10/10 split is generated (`--seed`, `--val_frac`, `--test_frac`).
- Empty (no-road) tiles are skipped — SAM-Road's loader skips empty graphs anyway.

Output (this is the SAM-Road dataset):

```
<out>/images/<id>.png          chosen RGB, 256×256 8-bit
<out>/road_masks/<id>.png      0/255, dilated by --road_buffer px
<out>/keypoint_masks/<id>.png  0/255 disks at intersections/endpoints
<out>/graphs_p/<id>.p          pickled adjacency dict {(row,col): [(row,col),…]}
<out>/data_split.json          {"train": [...], "validation": [...], "test": [...]}
```

Useful flags: `--limit N` (cap tiles, quick test), `--road_buffer PX` (road-mask
dilation radius, default 2; `0` = thin/verbatim), `--min_spur PX` (dead-end pruning
length, default 8; `0` = off), `--simplify PX` (polyline tolerance, default 2),
`--keypoint_radius PX` (default 3).

## Run on Kaggle (recommended — the source is already a Kaggle dataset)

`src/samroad/prep/kaggle_dependencies_s2roads.py` downloads the SAM checkpoint and
the raw dataset, runs the converter, and symlinks the result to `sam_road/s2roads_data`.
Pick the RGB variant with the `S2ROADS_IMAGES` env var.

In a Kaggle notebook (GPU on), after cloning the repo + submodule and installing deps
(`src/samroad/prep/prep_env_s2roads.bash` does both):

```bash
# convert true-colour images and train
S2ROADS_IMAGES=images_png /kaggle/working/InstaRoadPrototype/src/samroad/prep/prep_env_s2roads.bash
cd /kaggle/working/InstaRoadPrototype/sam_road
python train.py --config config/s2roads_256.yaml --precision 16
```

To compare against the enhanced images, re-run with `S2ROADS_IMAGES=images_enhanced_png`
(converted output is cached per variant; only the `s2roads_data` symlink flips), then
train again. Checkpoints (top-3 by `road_iou`) land in `lightning_logs/`.

## Run locally

```bash
# from sam_road/, with the project venv active (uv sync --extra samroad)
python s2roads/prepare_s2roads.py --src <kagglehub_root> --images images_png         --out ./s2roads_data
# ...or the enhanced variant:
python s2roads/prepare_s2roads.py --src <kagglehub_root> --images images_enhanced_png --out ./s2roads_data
python train.py --config config/s2roads_256.yaml --precision 16
```

The loader expects the dataset at `./s2roads_data` (relative to `sam_road/`) and the
SAM checkpoint at `SAM_CKPT_PATH`. `sknw` + `scikit-image` (converter only) are in the
`samroad` optional-dependency set.

### Keypoint head collapsing to 0?

The keypoint (intersection) channel is extremely sparse on this dataset: disks at
degree≠2 nodes are ~0.1% of pixels (median), and ~30% of tiles have **no** interior
intersection at all (just through-roads). `model.py` puts the keypoint and road
channels through one **unweighted** `BCEWithLogitsLoss`, so with `FOCAL_LOSS: False`
the keypoint logits collapse to all-negative (predict 0) — that is near-optimal BCE
for a 0.1%-positive channel. The road channel is fine because the `--road_buffer`
dilation made it ~3-4% positive.

Mitigations (the first two are the defaults now):
1. **`FOCAL_LOSS: True`** in `config/s2roads_256.yaml` — focal (γ=2) down-weights easy
   negatives so the rare keypoints still drive gradient. Primary fix.
2. **`--keypoint_radius 5`** (converter default) — bigger disks ≈ 2.4× the positive
   area vs radius 3; raise to 7 if the head is still weak. Requires re-converting.
3. Still collapsing? Give the keypoint channel its own positive weight — pass
   `pos_weight=torch.tensor([w_kp, w_road])` (e.g. `[20., 1.]`) to the
   `BCEWithLogitsLoss` in `model.py` (the stacked channel is the last dim), or raise
   focal `alpha` toward 0.5–0.75.

The ~30% no-intersection tiles are genuine (rural through-roads); they correctly
supervise an all-zero keypoint map and focal keeps them from dominating — don't
"fix" them by relaxing the border exclusion (that just invents fake keypoints at
roads cut by the tile edge).

### Notes
- `config/s2roads_256.yaml`: `BATCH_SIZE: 16`, `TRAIN_EPOCHS: 30`, `vit_b`. A 16 GB
  T4/P100 fits batch 16 at 256². Drop to 8 on OOM.
- Graphs are skeletonised from masks, so topology is approximate (a thick blob or a
  gap in the mask becomes a spurious junction or a break). `--min_spur` trims the worst
  noise. This is the price of having no road vectors — unlike `s2rosa`, whose topology
  comes from Overture connectors.
- Inference: `python inferencer.py --config config/s2roads_256.yaml --checkpoint <ckpt>`.
