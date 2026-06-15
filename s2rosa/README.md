# Training SAM-Road on S2-ROSA (Kaggle)

SAM-Road needs, per tile: an RGB image, a road mask, a keypoint mask, and a road
**graph** (it predicts a keypoint heatmap + topology, not just segmentation). S2-ROSA
stores per-zone Cloud-Optimized GeoTIFFs + Overture road vectors. `prepare_s2rosa.py`
converts S2-ROSA into the PNG + pickle layout SAM-Road's loader already understands,
so training on Kaggle needs no rasterio / shapely / Overture parsing — only the
lightweight converted dataset.

**Each 256×256 S2-ROSA patch becomes one SAM-Road tile** (`IMAGE_SIZE=PATCH_SIZE=256`,
margin 0 — the sampler uses the whole patch). Splits come straight from S2-ROSA's
`split_set` column.

## Step 1 — convert locally (where the COGs live)

```bash
# from the sam_road/ directory, with the project venv active
python s2rosa/prepare_s2rosa.py \
    --src /Volumes/MacOSFiles/S2ROSA \
    --out /Volumes/MacOSFiles/S2ROSA_samroad
# ~5 min for all 20 zones
```

Output layout (this is the SAM-Road dataset):

```
S2ROSA_samroad/
  images/<zone>_<row>_<col>.png         8-bit true-colour RGB (B4,B3,B2), p2-p98 stretch
  road_masks/<zone>_<row>_<col>.png      0/255, reused from the S2-ROSA raster mask
  keypoint_masks/<zone>_<row>_<col>.png  0/255 disks at graph intersections/endpoints
  graphs_p/<zone>_<row>_<col>.p          pickled adjacency dict {(row,col): [(row,col),...]}
  data_split.json                        {"train": [...], "validation": [...], "test": [...]}
```

What the converter does:
- **RGB**: bands 1,2,3 (= B4/B3/B2 true colour) → per-image 2–98 percentile stretch → 8-bit.
- **Graph**: Overture LineStrings → adjacency dict. Each segment is split at its
  Overture **connectors** so mid-segment junctions (T-junctions) become real nodes;
  shared connectors snap to the same pixel → correct topology. Coords are integer
  (row, col) pixels inside the patch (SAM-Road's `coord_transform v[:, ::-1]` makes them x,y).
- **Class filter**: keeps only drivable classes (`motorway…residential, living_street`).
  Footways / paths / steps / service / parking aisles are dropped — they are sub-pixel at
  10 m/px, so supervising them would teach the model to hallucinate invisible features.
  Override with `--classes a,b,c`.
- **Keypoints**: graph nodes with degree ≠ 2 (intersections + dead-ends), excluding the
  patch border (roads cut by the window edge are not real keypoints).
- **Road mask**: reused verbatim from the S2-ROSA raster mask (the same target the U-Net
  trained on; it already corresponds to the vehicular network).
- Empty patches and partial edge blocks (< 256²) are skipped.

Useful flags: `--zones CapeTown,Durban` and `--limit N` (patches per zone) for quick tests.

## Step 2 — upload to Kaggle

Create two Kaggle datasets:
1. **s2rosa-samroad** — the entire `S2ROSA_samroad/` folder from Step 1.
2. **sam-vit-b** — the SAM ViT-B checkpoint:
   `wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth`

Push the `sam_road/` source itself either as a third dataset / Kaggle GitHub import,
or paste it into the notebook's working dir.

## Step 3 — train (Kaggle notebook, GPU on)

```python
import os, sys
# 1. SAM-Road source -> /kaggle/working/sam_road  (git clone or a code dataset)
%cd /kaggle/working/sam_road

# 2. wire up the converted dataset and the SAM checkpoint via symlinks
os.makedirs("sam_ckpts", exist_ok=True)
if not os.path.exists("s2rosa"):
    os.symlink("/kaggle/input/s2rosa-samroad", "s2rosa")              # -> ./s2rosa/{images,...,data_split.json}
if not os.path.exists("sam_ckpts/sam_vit_b_01ec64.pth"):
    os.symlink("/kaggle/input/sam-vit-b/sam_vit_b_01ec64.pth",
               "sam_ckpts/sam_vit_b_01ec64.pth")

# 3. deps not already in the Kaggle image (sam/segment_anything is vendored in-repo)
!pip install -q lightning pytorch_lightning addict python-igraph rtree

# 4. train (no wandb account needed)
os.environ["WANDB_DISABLED"] = "1"
!python train.py --config config/s2rosa_256.yaml --precision 16
```

Checkpoints (top-3 by `val_loss`) land in `lightning_logs/`. Download the best `.ckpt`.

### Notes / knobs
- `config/s2rosa_256.yaml`: `BATCH_SIZE: 16`, `TRAIN_EPOCHS: 30`, `vit_b`. A 16 GB
  T4/P100 fits batch 16 at 256². Drop to 8 if you hit OOM, raise on an A100.
- The loader expects the dataset at `./s2rosa` (relative to `sam_road/`) and the checkpoint
  at the path in `SAM_CKPT_PATH`. The symlinks above satisfy both.
- `rasterio` / `shapely` are **only** needed by the converter (Step 1, local), never on Kaggle.
- To finetune without SAM weights, set `NO_SAM: True` in the config (experimental).
- Inference: `python inferencer.py` — review its dataset/threshold args; the `s2rosa`
  branch uses the same paths. `*_THRESHOLD` in the config are on the 0–1 sigmoid scale.
