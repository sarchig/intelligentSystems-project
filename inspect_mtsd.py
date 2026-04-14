import json
import random
import math
from pathlib import Path

import matplotlib
matplotlib.use("Qt5Agg")  # Force an interactive backend for pop-up windows

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


# ============================================================
# Configuration (updated to your NEW folder structure)
# ============================================================

PROJECT_ROOT = Path(".")

# Root folder you showed:
# data/raw/mtsd_extracted/
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "mtsd_extracted"

# Annotation folder (same as before, but now under Annotation/)
DATASET_ROOT = RAW_ROOT / "Annotation" / "mtsd_v2_fully_annotated"
ANN_DIR = DATASET_ROOT / "annotations"

# (Not used for loading images in "Recommendation A", but keep for checks)
SPLITS_DIR = DATASET_ROOT / "splits"

# NEW: unified images folder
DATASET_IMAGES_DIR = RAW_ROOT / "Dataset" / "images"

IMAGE_EXT = ".jpg"

# Exclude label
EXCLUDE_LABEL = "other-sign"

# How many keys to sample for file existence checks
MATCH_SAMPLE_N = 300

# Total number of examples to visualize
PLOT_SAMPLES_N = 6

# Crop panel options
SHOW_CROPS = True

# Bounding box visualization options
BOX_COLOR = "red"
BOX_LINEWIDTH = 2.5
BOX_FILL = False  # Keep the box interior unfilled

# Crop grid layout: number of columns in the crop panel
CROP_GRID_COLS = 6

# Crop label rendering
CROP_LABEL_FONTSIZE = 8


# ============================================================
# Helper functions
# ============================================================

def list_annotation_keys() -> list[str]:
    """
    List all available annotation keys from ANN_DIR.
    Each annotation file is <key>.json.
    """
    if not ANN_DIR.exists():
        raise FileNotFoundError(f"Annotation directory not found: {ANN_DIR}")

    keys = []
    for p in ANN_DIR.glob("*.json"):
        keys.append(p.stem)
    return keys


def locate_image_path(image_key: str) -> Path | None:
    """
    Locate the image file path for a given image_key in the NEW unified images folder:
      Dataset/images/<key>.jpg
    """
    p = DATASET_IMAGES_DIR / f"{image_key}{IMAGE_EXT}"
    return p if p.exists() else None


def locate_annotation_path(image_key: str) -> Path | None:
    """
    Locate the annotation JSON for a given image_key:
      annotations/<key>.json
    """
    p = ANN_DIR / f"{image_key}.json"
    return p if p.exists() else None


def load_json(path: Path) -> dict:
    """Load a JSON file and return a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_annotation_schema(anno: dict) -> dict:
    """
    Summarize important schema fields per README:
      - width, height, ispano, objects
    """
    return {
        "width": anno.get("width"),
        "height": anno.get("height"),
        "ispano": anno.get("ispano"),
        "objects_count": len(anno.get("objects", [])) if isinstance(anno.get("objects", []), list) else None
    }


def extract_objects(anno: dict) -> list[dict]:
    """Return the list of object dicts from anno['objects']."""
    objs = anno.get("objects", [])
    if not isinstance(objs, list):
        return []
    return [o for o in objs if isinstance(o, dict)]


def extract_bbox(obj: dict):
    """
    Extract bbox from object dict:
      obj['bbox']['xmin'], ['ymin'], ['xmax'], ['ymax']
    Also includes panorama cross-boundary dict if present.
    """
    bbox = obj.get("bbox")
    if not isinstance(bbox, dict):
        return None

    if all(k in bbox for k in ["xmin", "ymin", "xmax", "ymax"]):
        xmin = float(bbox["xmin"])
        ymin = float(bbox["ymin"])
        xmax = float(bbox["xmax"])
        ymax = float(bbox["ymax"])
        return xmin, ymin, xmax, ymax, bbox.get("cross_boundary")

    return None


def clamp_bbox(xmin, ymin, xmax, ymax, w, h):
    """Clamp bbox coordinates to image bounds and validate non-empty box."""
    xmin = max(0, min(xmin, w - 1))
    ymin = max(0, min(ymin, h - 1))
    xmax = max(0, min(xmax, w - 1))
    ymax = max(0, min(ymax, h - 1))
    if xmax <= xmin or ymax <= ymin:
        return None
    return xmin, ymin, xmax, ymax


def is_excluded_label(label: str) -> bool:
    """Return True if the label should be excluded."""
    return str(label).strip() == EXCLUDE_LABEL


# ============================================================
# Visualization
# ============================================================

def plot_image_and_all_crops_same_window(image_path: Path, anno: dict, title: str):
    """
    Show a single pop-up window that contains:
      - Top: original image with red bbox outlines only (no text, no fill)
      - Bottom: ALL cropped sign patches in a grid with class labels
    BUT:
      - Exclude 'other-sign' from BOTH bbox overlay and crops.
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    objs = extract_objects(anno)

    # Collect valid boxes and labels (excluding 'other-sign')
    boxes = []
    labels = []

    for obj in objs:
        # Exclude label early
        lbl = obj.get("label", "unknown")
        if is_excluded_label(lbl):
            continue

        out = extract_bbox(obj)
        if out is None:
            continue

        xmin, ymin, xmax, ymax, cross_boundary = out

        # Skip panorama cross-boundary case in this inspection view for simplicity
        if cross_boundary is not None and xmin > xmax:
            continue

        clamped = clamp_bbox(xmin, ymin, xmax, ymax, w, h)
        if clamped is None:
            continue

        boxes.append(clamped)
        labels.append(lbl)

    n_crops = len(boxes) if SHOW_CROPS else 0

    # Compute crop grid shape
    if SHOW_CROPS and n_crops > 0:
        cols = max(1, CROP_GRID_COLS)
        rows = int(math.ceil(n_crops / cols))
    else:
        cols = 1
        rows = 1

    # Make the figure size adaptive
    fig_w = 14
    fig_h = 6 + (rows * 2.4 if SHOW_CROPS and n_crops > 0 else 0)

    fig = plt.figure(figsize=(fig_w, fig_h))
    if SHOW_CROPS:
        gs = fig.add_gridspec(2, 1, height_ratios=[4, max(1.0, rows * 1.8)])
        ax_img = fig.add_subplot(gs[0])
        ax_crops_container = fig.add_subplot(gs[1])
    else:
        ax_img = fig.add_subplot(1, 1, 1)
        ax_crops_container = None

    # Top panel: full image + red bbox outlines only
    ax_img.imshow(img)
    ax_img.set_title(title)
    ax_img.axis("off")

    for (xmin, ymin, xmax, ymax) in boxes:
        rect = patches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=BOX_FILL,
            linewidth=BOX_LINEWIDTH,
            edgecolor=BOX_COLOR
        )
        ax_img.add_patch(rect)

    # Bottom panel: all crops grid with labels
    if SHOW_CROPS and ax_crops_container is not None:
        ax_crops_container.axis("off")

        if n_crops == 0:
            ax_crops_container.set_title(f"No valid crops found.")
        else:
            for idx, ((xmin, ymin, xmax, ymax), cls) in enumerate(zip(boxes, labels)):
                r = idx // cols
                c = idx % cols

                # Compute normalized position within the container axis
                cell_w = 1.0 / cols
                cell_h = 1.0 / rows

                left = c * cell_w + 0.01
                bottom = 1.0 - (r + 1) * cell_h + 0.06
                width = cell_w - 0.02
                height = cell_h - 0.10

                inset = ax_crops_container.inset_axes([left, bottom, width, height])

                crop = img.crop((xmin, ymin, xmax, ymax))
                inset.imshow(crop)

                # Show class label under/above crop as the subplot title
                inset.set_title(str(cls), fontsize=CROP_LABEL_FONTSIZE)

                inset.axis("off")

    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================

def main():
    # Basic directory checks
    print(f"[CHECK] RAW_ROOT exists = {RAW_ROOT.exists()} -> {RAW_ROOT}")
    print(f"[CHECK] DATASET_ROOT exists = {DATASET_ROOT.exists()} -> {DATASET_ROOT}")
    print(f"[CHECK] ANN_DIR exists = {ANN_DIR.exists()} -> {ANN_DIR}")
    print(f"[CHECK] SPLITS_DIR exists = {SPLITS_DIR.exists()} -> {SPLITS_DIR}")
    print(f"[CHECK] DATASET_IMAGES_DIR exists = {DATASET_IMAGES_DIR.exists()} -> {DATASET_IMAGES_DIR}")
    print(f"[INFO] Excluding label = '{EXCLUDE_LABEL}'")

    # Load annotation keys (Recommendation A: do not rely on train/val/test split files)
    keys_all = list_annotation_keys()
    print(f"[INFO] Total annotation keys found = {len(keys_all)}")

    # Matching check: sample keys and verify files exist
    print("\n[MATCH] Verifying image_key -> (image file, annotation file) existence...")
    sample = random.sample(keys_all, k=min(MATCH_SAMPLE_N, len(keys_all)))

    img_ok = 0
    ann_ok = 0
    both_ok = 0

    for k in sample:
        img_path = locate_image_path(k)
        ann_path = locate_annotation_path(k)

        if img_path is not None:
            img_ok += 1
        if ann_path is not None:
            ann_ok += 1
        if img_path is not None and ann_path is not None:
            both_ok += 1

    print(
        f"  - checked={len(sample)}, "
        f"images_found={img_ok}, annos_found={ann_ok}, both_found={both_ok}"
    )

    # Schema inspection: show top-level keys for a few random annotations
    print("\n[SCHEMA] Showing top-level keys for 5 random annotation JSON files...")
    schema_keys = random.sample(keys_all, k=min(5, len(keys_all)))
    for k in schema_keys:
        ann_path = locate_annotation_path(k)
        if ann_path is None:
            continue
        anno = load_json(ann_path)
        keys = list(anno.keys())
        print(f"  - {k}.json: keys={keys[:20]}{'...' if len(keys) > 20 else ''}")
        print(f"    summary={summarize_annotation_schema(anno)}")

    # Plot examples (random matched examples)
    print(f"\n[PLOT] Visualizing {PLOT_SAMPLES_N} random matched examples with bbox outlines and ALL crops (same window)...")
    plotted = 0

    # Candidate pool: random subset (to avoid scanning everything)
    subset = random.sample(keys_all, k=min(2500, len(keys_all)))
    random.shuffle(subset)

    for k in subset:
        img_path = locate_image_path(k)
        ann_path = locate_annotation_path(k)
        if img_path is None or ann_path is None:
            continue

        anno = load_json(ann_path)
        title = f"key={k} | objects={len(extract_objects(anno))}"
        plot_image_and_all_crops_same_window(img_path, anno, title)

        plotted += 1
        if plotted >= PLOT_SAMPLES_N:
            break

    if plotted == 0:
        print("[WARN] No matched samples found for plotting. Check folder structure.")
    else:
        print(f"[DONE] Plotted {plotted} examples.")

    print("\n[NEXT] Next step is to build a sign-crop classification dataset:")
    print("       - For each (image_key, object bbox), crop the sign region")
    print(f"       - EXCLUDE label '{EXCLUDE_LABEL}' (do not save these crops)")
    print("       - Save crops to disk and create a CSV (crop_path, label)")
    print("       - Create new train/val/test split yourself (Recommendation A)")
    print("       - Train multiple CNN backbones for traffic sign classification")


if __name__ == "__main__":
    main()
