"""Convert the Kaggle "Sentinel-2 Roads" dataset into the layout SAM-Road expects.

Source: kagglehub.dataset_download("sonisuyash/sentinel-2-roads-dataset"). The
download is wrapped in a numeric parent folder ("15765738") and every leaf folder
is doubled, e.g.:

    <root>/15765738/images_png/images_png/<id>.png            256x256 8-bit RGB
    <root>/15765738/images_enhanced_png/images_enhanced_png/<id>.png  contrast-stretched RGB
    <root>/15765738/masks_png/masks_png/<id>.png              256x256 0/255 binary road mask
    <root>/15765738/images_tif/...   <root>/15765738/masks_tif/...    (georeferenced, unused)

Unlike S2-ROSA, this dataset ships NO road vectors -- only raster masks. SAM-Road
needs a road *graph* (it predicts a keypoint heatmap + topology), so we recover the
graph from each binary mask by skeletonising it (scikit-image) and tracing the
1-px skeleton into a node/edge graph (sknw). Every 256x256 tile becomes one
SAM-Road tile (IMAGE_SIZE == PATCH_SIZE == 256, margin 0 -- the sampler uses the
whole tile), exactly like the s2rosa branch.

For every non-empty tile we emit:
    <out>/images/<id>.png          the chosen RGB (images_png or images_enhanced_png), copied verbatim
    <out>/road_masks/<id>.png      0/255 road mask, dilated by --road_buffer px (the raw
                                   masks_png roads are ~1 px wide, too thin a segmentation
                                   target; the graph/keypoints still come from the thin mask)
    <out>/keypoint_masks/<id>.png  0/255 disks at graph nodes with degree != 2
    <out>/graphs_p/<id>.p          pickled adjacency dict {(row,col): [(row,col), ...]}
    <out>/data_split.json          {"train": [...], "validation": [...], "test": [...]}

The adjacency dict is the sat2graph format SAM-Road's GraphLabelGenerator consumes
(integer (row, col) pixels inside the patch; coord_transform v[:, ::-1] turns them
into x=col, y=row). The dataset has no published split, so we make a deterministic
random one (seeded).

Run, once per image variant (the RGB source is the only thing that differs):
    python s2roads/prepare_s2roads.py --src <kagglehub_root> --images images_png          --out ./s2roads_data_plain
    python s2roads/prepare_s2roads.py --src <kagglehub_root> --images images_enhanced_png  --out ./s2roads_data_enhanced

then point ./s2roads_data at whichever you want to train (see config/s2roads_256.yaml).
"""
import argparse
import glob
import json
import os
import pickle
import random
import shutil
from collections import defaultdict

import cv2
import numpy as np
import sknw
from skimage.morphology import skeletonize

PATCH = 256              # tile size == SAM-Road PATCH_SIZE
KEYPOINT_RADIUS = 3      # matches sam_road/cityscale/generate_labels.py
SIMPLIFY_TOL_PX = 2.0    # approxPolyDP tolerance: drop near-collinear skeleton vertices
MIN_SPUR_PX = 8.0        # prune skeleton dead-end branches shorter than this (skeleton noise)
BORDER = 2               # nodes within BORDER px of the edge are cut roads, not real keypoints
ROAD_BUFFER_PX = 2       # dilate the thin raster road mask by this radius for the seg target
                         # (~10 m/px Sentinel-2: radius 2 -> ~5 px / ~50 m band)


def find_leaf_dir(root, name):
    """Locate the (possibly doubled / wrapper-nested) folder named `name` that
    directly holds the image files. Returns the dir that contains the pngs."""
    hits = glob.glob(os.path.join(root, "**", name, "*.png"), recursive=True)
    if not hits:
        raise FileNotFoundError(f"no '*.png' found under any '{name}/' below {root}")
    # All hits share the same parent dir (the leaf). Take it from the first hit.
    return os.path.dirname(hits[0])


def prune_spurs(g, min_spur):
    """Drop short dead-end (degree-1) branches in-place; these are skeletonisation
    noise (whiskers off thick road blobs), not real roads. Repeat to peel chains.
    Degree is recomputed live because removals change neighbours' degrees mid-pass."""
    if min_spur <= 0:
        return
    changed = True
    while changed:
        changed = False
        for n in list(g.nodes()):
            if n not in g:
                continue
            deg = g.degree(n)
            if deg == 0:
                g.remove_node(n)  # isolated speck left by skeletonisation
                changed = True
            elif deg == 1:
                nbr = next(iter(g[n]))
                # sknw stores the traced path length on the edge 'weight'
                if g[n][nbr].get("weight", 0.0) < min_spur:
                    g.remove_node(n)
                    changed = True


def mask_to_adjacency(mask, simplify_tol, min_spur):
    """Binary road mask -> sat2graph adjacency dict {(row,col): [(row,col), ...]}.

    Skeletonise -> trace to a node/edge graph (sknw) -> prune noise spurs -> chain
    each (simplified) edge polyline into consecutive integer-pixel edges. Coords are
    (row, col) inside the patch.
    """
    ske = skeletonize(mask > 0)
    g = sknw.build_sknw(ske)
    prune_spurs(g, min_spur)

    adj = defaultdict(set)
    for s, e in g.edges():
        pts = g[s][e]["pts"]  # [(row,col), ...] traced skeleton path (excludes nodes)
        poly = np.concatenate([g.nodes[s]["o"][None], pts, g.nodes[e]["o"][None]], axis=0)
        # approxPolyDP wants (x=col, y=row); simplify the polyline, then go back to (row,col)
        xy = poly[:, ::-1].astype(np.float32).reshape(-1, 1, 2)
        appr = cv2.approxPolyDP(xy, simplify_tol, False).reshape(-1, 2)
        pix = [(int(round(y)), int(round(x))) for x, y in appr]
        for a, b in zip(pix[:-1], pix[1:]):
            if a != b:
                adj[a].add(b)
                adj[b].add(a)
    return {k: list(v) for k, v in adj.items()}


def buffer_mask(mask, radius):
    """Thin binary road mask -> wider 0/255 band via morphological dilation (the
    segmentation training target). radius px; a circular (elliptical) structuring
    element approximates a buffer of that radius. radius 0 -> verbatim thin mask."""
    binary = (mask > 0).astype(np.uint8)
    if radius > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        binary = cv2.dilate(binary, k)
    return binary * 255


def draw_keypoints(adj, radius):
    """Disks at graph nodes with degree != 2 (intersections + dead-ends), excluding
    the patch border. Returns [PATCH, PATCH] uint8 0/255."""
    img = np.zeros((PATCH, PATCH), dtype=np.uint8)
    for (row, col), neighbors in adj.items():
        if len(neighbors) == 2:
            continue
        if row <= BORDER or row >= PATCH - 1 - BORDER or col <= BORDER or col >= PATCH - 1 - BORDER:
            continue
        cv2.circle(img, (col, row), radius, 255, -1)  # cv2 takes (x=col, y=row)
    return img


def main():
    ap = argparse.ArgumentParser(description="Convert Kaggle Sentinel-2 Roads to SAM-Road format.")
    ap.add_argument("--src", required=True,
                    help="dataset root (the kagglehub download dir; the '15765738' wrapper is found automatically)")
    ap.add_argument("--images", default="images_png",
                    choices=["images_png", "images_enhanced_png"],
                    help="which RGB variant to use as the input image")
    ap.add_argument("--out", required=True, help="output root (SAM-Road dataset dir)")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--test_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0, help="seed for the random train/val/test split")
    ap.add_argument("--simplify", type=float, default=SIMPLIFY_TOL_PX, help="polyline simplify tolerance (px)")
    ap.add_argument("--min_spur", type=float, default=MIN_SPUR_PX, help="prune dead-end branches shorter than this (px); 0=off")
    ap.add_argument("--keypoint_radius", type=int, default=KEYPOINT_RADIUS)
    ap.add_argument("--road_buffer", type=int, default=ROAD_BUFFER_PX,
                    help="dilate the road mask by this radius (px) for the seg target; 0 = thin/verbatim")
    ap.add_argument("--limit", type=int, default=0, help="max tiles processed (0 = no limit), for quick tests")
    args = ap.parse_args()

    img_dir = find_leaf_dir(args.src, args.images)
    mask_dir = find_leaf_dir(args.src, "masks_png")
    print(f"images: {img_dir}\nmasks : {mask_dir}")

    img_ids = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(img_dir, "*.png"))}
    mask_ids = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(mask_dir, "*.png"))}
    ids = sorted(img_ids & mask_ids, key=lambda s: int(s) if s.isdigit() else s)
    missing = len(img_ids ^ mask_ids)
    print(f"paired tiles: {len(ids)} (unpaired ids skipped: {missing})")

    for sub in ("images", "road_masks", "keypoint_masks", "graphs_p"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    # deterministic train/val/test split over the paired ids
    rng = random.Random(args.seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = int(n * args.test_frac)
    n_val = int(n * args.val_frac)
    test_set = set(shuffled[:n_test])
    val_set = set(shuffled[n_test:n_test + n_val])
    splits = {"train": [], "validation": [], "test": []}

    n_written = n_empty = 0
    for i, tid in enumerate(ids):
        if args.limit and n_written >= args.limit:
            break
        mask = cv2.imread(os.path.join(mask_dir, f"{tid}.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        if mask.shape[:2] != (PATCH, PATCH):
            mask = cv2.resize(mask, (PATCH, PATCH), interpolation=cv2.INTER_NEAREST)

        adj = mask_to_adjacency(mask, args.simplify, args.min_spur)
        if len(adj) == 0:
            n_empty += 1
            continue  # no roads -> SAM-Road's loader skips empty graphs anyway

        # image: copy the chosen RGB verbatim (already 256x256 8-bit). Re-encode through
        # cv2 only if it is not already a .png of the right size.
        src_img = os.path.join(img_dir, f"{tid}.png")
        dst_img = os.path.join(args.out, "images", f"{tid}.png")
        img = cv2.imread(src_img, cv2.IMREAD_COLOR)
        if img is not None and img.shape[:2] == (PATCH, PATCH):
            shutil.copyfile(src_img, dst_img)
        else:
            cv2.imwrite(dst_img, cv2.resize(img, (PATCH, PATCH)))

        # road mask: dilate the thin raster mask into a wider seg target (0/255)
        road = buffer_mask(mask, args.road_buffer)
        cv2.imwrite(os.path.join(args.out, "road_masks", f"{tid}.png"), road)

        cv2.imwrite(os.path.join(args.out, "keypoint_masks", f"{tid}.png"),
                    draw_keypoints(adj, args.keypoint_radius))

        with open(os.path.join(args.out, "graphs_p", f"{tid}.p"), "wb") as f:
            pickle.dump(adj, f)

        split = "test" if tid in test_set else "validation" if tid in val_set else "train"
        splits[split].append(tid)
        n_written += 1
        if n_written % 500 == 0:
            print(f"  ... {n_written} tiles written")

    with open(os.path.join(args.out, "data_split.json"), "w") as f:
        json.dump(splits, f, indent=2)

    print(f"\nDone. written={n_written} empty_skipped={n_empty}")
    print(f"split sizes: train={len(splits['train'])} val={len(splits['validation'])} test={len(splits['test'])}")
    print(f"output: {args.out}")


if __name__ == "__main__":
    main()
