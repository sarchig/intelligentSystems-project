#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
save_crops_csv.py

Build a sign-crop classification dataset from MTSD annotation JSON files.

Key changes (per your new plan):
- Recommendation A: Do NOT rely on original train/val/test split files.
  → Use ALL annotation keys and a unified images folder: data/raw/mtsd_extracted/Dataset/images
- Exclude label "other-sign" completely:
  → Do not save its crops
  → Do not write it into CSV
- Output crops organized by class only:
  out_dir/
    crops/<label>/<imagekey>_obj###_<label>.png
    crops.csv

Later, train_keras_signs.py will create NEW train/val/test splits from this CSV.
"""

import json
import csv
import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

from PIL import Image


# ============================================================
# Configuration (NEW folder structure + exclusion rule)
# ============================================================

PROJECT_ROOT = Path(".")

RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "mtsd_extracted"

# Annotations
DATASET_ROOT = RAW_ROOT / "Annotation" / "mtsd_v2_fully_annotated"
ANN_DIR = DATASET_ROOT / "annotations"

# Unified images folder (Recommendation A)
IMAGES_DIR = RAW_ROOT / "Dataset" / "images"

IMAGE_EXT = ".jpg"

# Exclude label
EXCLUDE_LABEL = "other-sign"


# ============================================================
# Helper functions
# ============================================================

def list_annotation_keys(ann_dir: Path) -> List[str]:
    """Return all annotation keys from ann_dir (each file is <key>.json)."""
    if not ann_dir.exists():
        raise FileNotFoundError(f"Annotation directory not found: {ann_dir}")
    return [p.stem for p in ann_dir.glob("*.json")]


def locate_image_path(images_dir: Path, image_key: str, image_ext: str) -> Optional[Path]:
    """Locate the image file path: Dataset/images/<key>.jpg"""
    p = images_dir / f"{image_key}{image_ext}"
    return p if p.exists() else None


def locate_annotation_path(ann_dir: Path, image_key: str) -> Optional[Path]:
    """Locate the annotation JSON file: annotations/<key>.json"""
    p = ann_dir / f"{image_key}.json"
    return p if p.exists() else None


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file and return a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_objects(anno: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the list of object dicts from anno['objects']."""
    objs = anno.get("objects", [])
    if not isinstance(objs, list):
        return []
    return [o for o in objs if isinstance(o, dict)]


def extract_bbox_xyxy(obj: Dict[str, Any]) -> Optional[Tuple[float, float, float, float, Any]]:
    """
    Extract bbox from object dict:
      obj['bbox']['xmin','ymin','xmax','ymax']
    Return (xmin, ymin, xmax, ymax, cross_boundary).
    """
    bbox = obj.get("bbox")
    if not isinstance(bbox, dict):
        return None

    required = ["xmin", "ymin", "xmax", "ymax"]
    if not all(k in bbox for k in required):
        return None

    xmin = float(bbox["xmin"])
    ymin = float(bbox["ymin"])
    xmax = float(bbox["xmax"])
    ymax = float(bbox["ymax"])
    cross_boundary = bbox.get("cross_boundary")

    return xmin, ymin, xmax, ymax, cross_boundary


def clamp_bbox_xyxy(xmin: float, ymin: float, xmax: float, ymax: float, w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
    """Clamp bbox to image bounds and validate non-empty box."""
    xmin = max(0.0, min(xmin, float(w - 1)))
    ymin = max(0.0, min(ymin, float(h - 1)))
    xmax = max(0.0, min(xmax, float(w - 1)))
    ymax = max(0.0, min(ymax, float(h - 1)))
    if xmax <= xmin or ymax <= ymin:
        return None
    return int(xmin), int(ymin), int(xmax), int(ymax)


def sanitize_label_for_filename(label: str) -> str:
    """Make label safe for filenames."""
    label = str(label).strip()
    label = re.sub(r"\s+", "_", label)
    label = re.sub(r"[^A-Za-z0-9_\-\.]", "_", label)
    return label if label else "unknown"


def is_excluded_label(label: str) -> bool:
    """Return True if this label should be excluded."""
    return str(label).strip() == EXCLUDE_LABEL


# ============================================================
# Core: save crops + write CSV (Recommendation A)
# ============================================================

def build_mtsd_crop_dataset(
    ann_dir: Path,
    images_dir: Path,
    out_dir: Path,
    image_ext: str = ".jpg",
    skip_cross_boundary: bool = True,
    min_crop_size_px: int = 2,
    max_images_total: Optional[int] = None,
) -> Path:
    """
    Build crops and write a CSV.

    Output:
      out_dir/
        crops/<label>/<imagekey>_obj###_<label>.png
        crops.csv

    CSV columns:
      crop_path, label, image_key, image_path, ann_path, obj_index,
      xmin, ymin, xmax, ymax, width, height
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    crops_root = out_dir / "crops"
    crops_root.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "crops.csv"

    fieldnames = [
        "crop_path",
        "label",
        "image_key",
        "image_path",
        "ann_path",
        "obj_index",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "width",
        "height",
    ]

    keys_all = list_annotation_keys(ann_dir)
    if max_images_total is not None:
        keys_all = keys_all[:max_images_total]

    print(f"[INFO] Total annotation keys to process = {len(keys_all)}")
    print(f"[INFO] Excluding label = '{EXCLUDE_LABEL}'")
    print(f"[INFO] Images dir = {images_dir}")
    print(f"[INFO] Annotations dir = {ann_dir}")
    print(f"[INFO] Output dir = {out_dir}")

    saved_crops = 0
    skipped_excluded = 0
    skipped_small = 0
    skipped_missing = 0
    skipped_cross_boundary = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()

        for idx_key, image_key in enumerate(keys_all, 1):
            img_path = locate_image_path(images_dir, image_key, image_ext)
            ann_path = locate_annotation_path(ann_dir, image_key)

            if img_path is None or ann_path is None:
                skipped_missing += 1
                continue

            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                skipped_missing += 1
                continue

            w, h = img.size

            try:
                anno = load_json(ann_path)
            except Exception:
                skipped_missing += 1
                continue

            objects = extract_objects(anno)

            for obj_index, obj in enumerate(objects):
                label = obj.get("label", "unknown")
                label_str = str(label)

                # Exclude 'other-sign' completely
                if is_excluded_label(label_str):
                    skipped_excluded += 1
                    continue

                bbox_out = extract_bbox_xyxy(obj)
                if bbox_out is None:
                    continue

                xmin, ymin, xmax, ymax, cross_boundary = bbox_out

                # Skip panorama wrap-around
                if skip_cross_boundary and cross_boundary is not None and xmin > xmax:
                    skipped_cross_boundary += 1
                    continue

                clamped = clamp_bbox_xyxy(xmin, ymin, xmax, ymax, w, h)
                if clamped is None:
                    continue

                xmin_i, ymin_i, xmax_i, ymax_i = clamped
                bw = xmax_i - xmin_i
                bh = ymax_i - ymin_i

                if bw < min_crop_size_px or bh < min_crop_size_px:
                    skipped_small += 1
                    continue

                label_safe = sanitize_label_for_filename(label_str)

                crop = img.crop((xmin_i, ymin_i, xmax_i, ymax_i))

                # Save to crops/<label>/...
                label_dir = crops_root / label_safe
                label_dir.mkdir(parents=True, exist_ok=True)

                crop_filename = f"{image_key}_obj{obj_index:03d}_{label_safe}.png"
                crop_path = label_dir / crop_filename

                try:
                    crop.save(crop_path)
                except Exception:
                    continue

                writer.writerow({
                    "crop_path": str(crop_path),
                    "label": label_str,
                    "image_key": image_key,
                    "image_path": str(img_path),
                    "ann_path": str(ann_path),
                    "obj_index": obj_index,
                    "xmin": xmin_i,
                    "ymin": ymin_i,
                    "xmax": xmax_i,
                    "ymax": ymax_i,
                    "width": bw,
                    "height": bh,
                })

                saved_crops += 1

            if idx_key % 5000 == 0:
                print(f"[INFO] Processed keys: {idx_key}/{len(keys_all)} | saved_crops={saved_crops}")

    print("\n[DONE] Crop dataset build finished.")
    print(f"[DONE] CSV written to: {csv_path}")
    print(f"[DONE] Crops saved under: {crops_root}")
    print(f"[STATS] saved_crops={saved_crops}")
    print(f"[STATS] skipped_excluded(other-sign)={skipped_excluded}")
    print(f"[STATS] skipped_small={skipped_small}")
    print(f"[STATS] skipped_cross_boundary={skipped_cross_boundary}")
    print(f"[STATS] skipped_missing(image/anno/read)={skipped_missing}")

    return csv_path


# ============================================================
# CLI entry point
# ============================================================

def main() -> None:
    out_dir = PROJECT_ROOT / "data" / "processed" / "mtsd_sign_crops_no_other_sign"
    csv_path = build_mtsd_crop_dataset(
        ann_dir=ANN_DIR,
        images_dir=IMAGES_DIR,
        out_dir=out_dir,
        image_ext=IMAGE_EXT,
        skip_cross_boundary=True,
        min_crop_size_px=2,
        max_images_total=None,
    )
    print(f"[DONE] Crop dataset CSV written to: {csv_path}")


if __name__ == "__main__":
    main()
