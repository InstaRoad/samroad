"""Convert the S2-ROSA dataset into the directory layout SAM-Road expects.

S2-ROSA (produced by src/sentinel2data) stores, per zone:
    imagery/<zone>.tif            20-band float32 COG (band 1=B4/R, 2=B3/G, 3=B2/B)
    masks_raster/<zone>_mask.tif  1-band uint8 binary road mask COG
    masks_graph/<zone>_graphs.parquet  Overture road segments (WKB LineStrings in
                                       the COG's UTM CRS), tagged per patch via
                                       patch_row_id / patch_col_id
    metadata.parquet              one row per 256x256 patch (split_set, road_density, ...)

Each 256x256 patch becomes one SAM-Road "tile". For every non-empty patch we emit:
    <out>/images/<id>.png          8-bit true-colour RGB (B4,B3,B2), per-image p2-p98 stretch
    <out>/road_masks/<id>.png      0/255 road mask (reused from the S2-ROSA raster)
    <out>/keypoint_masks/<id>.png  0/255 disks at graph nodes with degree != 2
    <out>/graphs_p/<id>.p          pickled adjacency dict {(row,col): [(row,col), ...]}
    <out>/data_split.json          {"train": [...], "validation": [...], "test": [...]}

where <id> = "<zone>_<patch_row_id>_<patch_col_id>".

The adjacency dict is the sat2graph format SAM-Road's GraphLabelGenerator consumes
(coords are integer (row, col) pixels inside the patch; coord_transform v[:, ::-1]
turns them into (x=col, y=row)). Topology comes for free from Overture: connected
segments share an exact connector vertex, so snapping vertices to integer pixels and
merging coincident ones reconstructs the graph.

Run locally (where the COGs live), then upload <out> as a Kaggle dataset:
    python s2rosa/prepare_s2rosa.py --src /Volumes/MacOSFiles/S2ROSA --out /Volumes/MacOSFiles/S2ROSA_samroad
"""
import argparse
import json
import os
import pickle
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from shapely import wkb
from shapely.geometry import box, LineString
from shapely.ops import substring

PATCH = 256          # S2-ROSA internal block size == SAM-Road tile size
RGB_BANDS = (1, 2, 3)  # B4, B3, B2 -> true-colour R, G, B
# Overture road classes kept for the graph. Footway/path/steps/cycleway/service etc.
# are dropped: they are sub-pixel at 10 m/px, so supervising them teaches the model
# to hallucinate invisible features. The raster road mask is left untouched.
DRIVABLE_CLASSES = {
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "residential", "unclassified", "living_street",
}
KEYPOINT_RADIUS = 3  # matches sam_road/cityscale/generate_labels.py
STRETCH_LO, STRETCH_HI = 2, 98  # per-image percentile stretch for the 8-bit RGB
SIMPLIFY_TOL_PX = 1.0  # drop sub-pixel Overture curve vertices (10 m/px) before snapping
BORDER = 2  # nodes within BORDER px of the patch edge are cut roads, not real keypoints


def stretch_to_uint8(bands):
    """bands: [3, H, W] float32 reflectance -> [H, W, 3] uint8 RGB, per-channel p2-p98."""
    out = np.zeros((bands.shape[1], bands.shape[2], 3), dtype=np.uint8)
    for c in range(3):
        b = np.nan_to_num(bands[c], nan=0.0, posinf=0.0, neginf=0.0)
        lo, hi = np.percentile(b, STRETCH_LO), np.percentile(b, STRETCH_HI)
        if hi <= lo:
            hi = lo + 1e-6
        out[..., c] = np.clip((b - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    return out


def _connector_ats(connectors):
    """Sorted unique parametric positions (incl. endpoints 0 and 1) where the
    segment is split into junction-to-junction pieces."""
    ats = {0.0, 1.0}
    if connectors is not None:
        for c in connectors:
            a = float(c["at"])
            if 0.0 <= a <= 1.0:
                ats.add(a)
    return sorted(ats)


def segments_to_adjacency(records, inv_transform, row_off, col_off):
    """Overture road segments -> sat2graph adjacency dict in patch-local pixels.

    records: list of (geom, connectors) where geom is a shapely LineString in the
        COG's CRS and connectors is the Overture connectors array (or None).
    inv_transform: ~src.transform (maps x,y -> col,row).
    row_off, col_off: top-left of the patch window in full-COG pixels.

    Topology is recovered from connectors: each segment is split at its connector
    positions so a junction that lands mid-segment (e.g. a T-junction) becomes an
    explicit node. Shared connectors are the same physical point, so after snapping
    to the integer pixel grid they merge -> correct adjacency. Edges crossing the
    window are clipped to the border (the stub end becomes a degree-1 node).

    Returns {(row, col): [(row, col), ...]} with integer pixel coords inside the patch.
    """
    ia, ib, ic = inv_transform.a, inv_transform.b, inv_transform.c
    id_, ie, if_ = inv_transform.d, inv_transform.e, inv_transform.f
    patch_box = box(0, 0, PATCH, PATCH)
    adj = defaultdict(set)

    def to_pixel(coords):
        # [(x_utm, y_utm), ...] -> [(col, row), ...] patch-local floats
        return [(ia * x + ib * y + ic - col_off, id_ * x + ie * y + if_ - row_off)
                for x, y in coords]

    def add_piece(piece):
        # piece: shapely LineString in patch-local (col, row); clip + snap + add edges
        clipped = piece.intersection(patch_box)
        if clipped.is_empty:
            return
        parts = clipped.geoms if clipped.geom_type.startswith("Multi") else [clipped]
        for part in parts:
            if part.geom_type != "LineString":
                continue
            part = part.simplify(SIMPLIFY_TOL_PX)  # remove redundant curve vertices
            pts = [(int(round(r)), int(round(c))) for c, r in part.coords]  # (row, col)
            for a, b in zip(pts[:-1], pts[1:]):
                if a != b:
                    adj[a].add(b)
                    adj[b].add(a)

    for geom, connectors in records:
        if geom is None or geom.is_empty:
            continue
        geoms = geom.geoms if geom.geom_type == "MultiLineString" else [geom]
        for g in geoms:
            if g.geom_type != "LineString" or g.length == 0:
                continue
            ats = _connector_ats(connectors) if len(geoms) == 1 else [0.0, 1.0]
            for a0, a1 in zip(ats[:-1], ats[1:]):
                if a1 - a0 < 1e-9:
                    continue
                piece = substring(g, a0, a1, normalized=True)  # junction-to-junction sub-polyline
                if piece.geom_type != "LineString" or piece.length == 0:
                    continue
                add_piece(LineString(to_pixel(piece.coords)))

    return {k: list(v) for k, v in adj.items()}


def draw_keypoints(adj):
    """Disks (radius KEYPOINT_RADIUS) at graph nodes with degree != 2. Returns [PATCH,PATCH] uint8."""
    img = np.zeros((PATCH, PATCH), dtype=np.uint8)
    for (row, col), neighbors in adj.items():
        if len(neighbors) == 2:
            continue
        # skip nodes hard against the patch edge: these are roads cut by the
        # window boundary, not genuine intersections/endpoints.
        if row <= BORDER or row >= PATCH - 1 - BORDER or col <= BORDER or col >= PATCH - 1 - BORDER:
            continue
        cv2.circle(img, (col, row), KEYPOINT_RADIUS, 255, -1)  # cv2 takes (x=col, y=row)
    return img


def main():
    ap = argparse.ArgumentParser(description="Convert S2-ROSA to SAM-Road format.")
    ap.add_argument("--src", default="/Volumes/MacOSFiles/S2ROSA", help="S2-ROSA root")
    ap.add_argument("--out", default="/Volumes/MacOSFiles/S2ROSA_samroad", help="output root")
    ap.add_argument("--zones", default=None, help="comma-separated zone names (default: all)")
    ap.add_argument("--limit", type=int, default=0, help="max patches per zone (0 = no limit)")
    ap.add_argument("--classes", default=None,
                    help="comma-separated Overture road classes to keep (default: drivable set)")
    args = ap.parse_args()
    keep_classes = set(args.classes.split(",")) if args.classes else DRIVABLE_CLASSES

    src, out = args.src, args.out
    for sub in ("images", "road_masks", "keypoint_masks", "graphs_p"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    meta = pd.read_parquet(os.path.join(src, "metadata.parquet"))
    zones = args.zones.split(",") if args.zones else sorted(meta["zone_name"].unique())
    splits = {"train": [], "validation": [], "test": []}
    split_map = {"train": "train", "val": "validation", "validation": "validation", "test": "test"}
    n_written = n_empty = n_edge = 0

    for zone in zones:
        zmeta = meta[meta["zone_name"] == zone]
        if len(zmeta) == 0:
            print(f"[skip] {zone}: not in metadata")
            continue
        img_path = os.path.join(src, zmeta.iloc[0]["tile_path"])
        mask_path = os.path.join(src, zmeta.iloc[0]["mask_raster_path"])
        graph_path = os.path.join(src, zmeta.iloc[0]["mask_graph_path"])

        gdf = pd.read_parquet(graph_path)
        gdf = gdf[gdf["class"].isin(keep_classes)]  # drop non-vehicular ways
        # group Overture segments by patch; decode WKB lazily per group
        groups = {key: grp for key, grp in gdf.groupby(["patch_row_id", "patch_col_id"])}

        written_this_zone = 0
        with rasterio.open(img_path) as img_src, rasterio.open(mask_path) as mask_src:
            inv = ~img_src.transform
            W, H = img_src.width, img_src.height
            for _, prow in zmeta.iterrows():
                if args.limit and written_this_zone >= args.limit:
                    break
                pr, pc = int(prow["patch_row_id"]), int(prow["patch_col_id"])
                row_off, col_off = pr * PATCH, pc * PATCH
                # drop partial edge blocks (keep only full 256x256 tiles)
                if col_off + PATCH > W or row_off + PATCH > H:
                    n_edge += 1
                    continue

                seg = groups.get((pr, pc))
                if seg is None or len(seg) == 0:
                    n_empty += 1
                    continue
                records = [(wkb.loads(bytes(g)), c)
                           for g, c in zip(seg["geometry"].values, seg["connectors"].values)]
                adj = segments_to_adjacency(records, inv, row_off, col_off)
                if len(adj) == 0:
                    n_empty += 1
                    continue

                tile_id = f"{zone}_{pr}_{pc}"
                win = Window(col_off, row_off, PATCH, PATCH)

                rgb = stretch_to_uint8(img_src.read(RGB_BANDS, window=win).astype(np.float32))
                cv2.imwrite(os.path.join(out, "images", f"{tile_id}.png"), rgb[..., ::-1])  # RGB->BGR for cv2

                road = (mask_src.read(1, window=win) > 0).astype(np.uint8) * 255
                cv2.imwrite(os.path.join(out, "road_masks", f"{tile_id}.png"), road)

                cv2.imwrite(os.path.join(out, "keypoint_masks", f"{tile_id}.png"), draw_keypoints(adj))

                with open(os.path.join(out, "graphs_p", f"{tile_id}.p"), "wb") as f:
                    pickle.dump(adj, f)

                split = split_map.get(str(prow["split_set"]).lower())
                if split:
                    splits[split].append(tile_id)
                written_this_zone += 1
                n_written += 1

        print(f"[ok]   {zone}: wrote {written_this_zone} patches")

    with open(os.path.join(out, "data_split.json"), "w") as f:
        json.dump(splits, f, indent=2)

    print(f"\nDone. written={n_written} empty_skipped={n_empty} edge_skipped={n_edge}")
    print(f"split sizes: train={len(splits['train'])} val={len(splits['validation'])} test={len(splits['test'])}")
    print(f"output: {out}")


if __name__ == "__main__":
    main()
