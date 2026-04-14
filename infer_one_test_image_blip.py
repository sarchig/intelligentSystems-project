#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
infer_one_test_image_blip.py

WSL-friendly demo viewer (SEQUENTIAL, blocking):
- Render figures with Matplotlib Agg (no WSL GUI).
- Save PNG files.
- Open PNG in a blocking Windows viewer (PowerShell WinForms),
  so the next image opens only after the current window is closed.

Figures:
1) Selected Input Image
2) Classification Result
3) Final Results (Classification + Context-Aware Scene Description)

Notes:
- No confidence shown in figures.
- Class names in descriptions are wrapped in single quotes.
- Alternative labels are narrative sentences.
- All comments are in English.
"""

from __future__ import annotations

import json
import math
import random
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Use file rendering only (no WSL GUI windows)

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np
import pandas as pd
from PIL import Image

import tensorflow as tf
from tensorflow import keras


# ============================================================
# Configuration
# ============================================================

BEST_MODEL_DIR = Path("outputs/train_runs/best_performance_from_validation")

MODEL_PATH = BEST_MODEL_DIR / "best_model.keras"
CLASS_NAMES_PATH = BEST_MODEL_DIR / "class_names.json"
SPLIT_MANIFEST_PATH = BEST_MODEL_DIR / "split_manifest.csv"

OUTPUT_DIR = BEST_MODEL_DIR / "demo_one_random_test_image"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (224, 224)
CONF_THRESH = 0.0

NEAR_THRESH = 0.08
FAR_THRESH = 0.02

TOPK_FOR_TEXT = 3
CROP_GRID_COLS = 6

BOX_COLOR = "red"
BOX_LINEWIDTH = 2.5
BOX_FILL = False

TITLE_FONTSIZE = 20
CROP_LABEL_FONTSIZE = 14
DESC_FONTSIZE = 14

BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"


# ============================================================
# TensorFlow GPU configuration
# ============================================================

def enable_memory_growth() -> None:
    """Enable GPU memory growth to reduce out-of-memory issues."""
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
    except Exception as e:
        print("[WARN] Could not enable memory growth:", e)


# ============================================================
# Windows blocking image viewer (PowerShell WinForms)
# ============================================================

def open_png_in_windows_blocking(png_path: Path, window_title: str = "Result") -> None:
    """
    Open an image in a blocking Windows window (close to continue).
    This uses PowerShell + WinForms and blocks until the window is closed.
    """
    try:
        win_path = subprocess.check_output(["wslpath", "-w", str(png_path)]).decode().strip()
    except Exception as e:
        print(f"[WARN] wslpath failed for {png_path}: {e}")
        return

    # PowerShell script to show a WinForms window with the image and block until closed
    ps_script = rf"""
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$form = New-Object System.Windows.Forms.Form
$form.Text = "{window_title}"
$form.StartPosition = "CenterScreen"
$form.Width = 1200
$form.Height = 800

$pb = New-Object System.Windows.Forms.PictureBox
$pb.Dock = "Fill"
$pb.SizeMode = "Zoom"

$img = [System.Drawing.Image]::FromFile("{win_path}")
$pb.Image = $img

$form.Controls.Add($pb)

# Ensure resources are released on close
$form.add_FormClosed({{
    $pb.Image.Dispose()
    $img.Dispose()
}})

[void]$form.ShowDialog()
"""

    subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_script],
        check=False
    )


def save_figure(fig, save_path: Path, dpi: int = 250) -> None:
    """Save a matplotlib figure to a PNG file."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    print(f"[SAVED] {save_path}")


# ============================================================
# Helper functions
# ============================================================

def load_class_names(path: Path) -> list[str]:
    """Load class names from class_names.json."""
    return json.loads(path.read_text(encoding="utf-8"))


def clamp_bbox(xmin, ymin, xmax, ymax, w, h):
    """Clamp bounding box coordinates to image boundaries."""
    xmin = max(0, min(int(xmin), w - 1))
    ymin = max(0, min(int(ymin), h - 1))
    xmax = max(0, min(int(xmax), w - 1))
    ymax = max(0, min(int(ymax), h - 1))
    if xmax <= xmin or ymax <= ymin:
        return None
    return xmin, ymin, xmax, ymax


def preprocess_crop_for_model(crop_pil: Image.Image) -> np.ndarray:
    """Preprocess a crop image for model inference."""
    crop = crop_pil.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(crop).astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def infer_relative_size(bbox, img_w, img_h) -> str:
    """Infer relative size (near / medium / far) from bbox area ratio."""
    x1, y1, x2, y2 = bbox
    bbox_area = (x2 - x1) * (y2 - y1)
    img_area = img_w * img_h
    ratio = bbox_area / img_area if img_area > 0 else 0.0
    if ratio >= NEAR_THRESH:
        return "near"
    if ratio <= FAR_THRESH:
        return "far"
    return "medium"


def infer_position(bbox, img_w) -> str:
    """Infer horizontal position (left / center / right) of the bounding box."""
    x1, _, x2, _ = bbox
    cx = (x1 + x2) / 2.0
    if cx < img_w / 3:
        return "left"
    if cx < 2 * img_w / 3:
        return "center"
    return "right"


def blip_caption(full_image_pil: Image.Image) -> str:
    """Generate a scene caption using BLIP."""
    from transformers import BlipProcessor, BlipForConditionalGeneration
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
    model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_NAME).to(device)

    inputs = processor(images=full_image_pil.convert("RGB"), return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=30)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption.strip()


def build_context_description(scene_caption: str, sign_descs: list[str], alt_sentences: list[str]) -> str:
    """Build a narrative context-aware scene description (labels quoted)."""
    n = len(sign_descs)
    if n == 0:
        base = f"Scene: {scene_caption}. No traffic signs were detected."
    elif n == 1:
        base = f"Scene: {scene_caption}. One traffic sign was detected: {sign_descs[0]}."
    else:
        base = f"Scene: {scene_caption}. {n} traffic signs were detected: " + "; ".join(sign_descs) + "."

    if alt_sentences:
        base += " " + " ".join(alt_sentences)
    return base


# ============================================================
# Plotting (save only)
# ============================================================

def make_figure_input(full_img: Image.Image) -> plt.Figure:
    """Create Figure 1."""
    fig = plt.figure(figsize=(10, 6))
    plt.imshow(full_img)
    plt.title("Selected Input Image", fontsize=TITLE_FONTSIZE)
    plt.axis("off")
    plt.tight_layout()
    return fig


def make_figure_panel(full_img: Image.Image, preds: list[dict], title: str, description_text: str | None = None) -> plt.Figure:
    """Create Figure 2/3."""
    n_crops = len(preds)
    cols = max(1, CROP_GRID_COLS)
    rows = max(1, math.ceil(n_crops / cols))

    fig_h = 6 + rows * 2.2 + (1.8 if description_text else 0)
    fig = plt.figure(figsize=(14, fig_h))

    if description_text:
        gs = fig.add_gridspec(3, 1, height_ratios=[4, rows * 1.8, 1.2])
        ax_img, ax_crops, ax_desc = [fig.add_subplot(gs[i]) for i in range(3)]
    else:
        gs = fig.add_gridspec(2, 1, height_ratios=[4, rows * 1.8])
        ax_img, ax_crops = [fig.add_subplot(gs[i]) for i in range(2)]
        ax_desc = None

    ax_img.imshow(full_img)
    ax_img.set_title(title, fontsize=TITLE_FONTSIZE)
    ax_img.axis("off")

    for p in preds:
        x1, y1, x2, y2 = p["bbox"]
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=BOX_FILL,
            linewidth=BOX_LINEWIDTH,
            edgecolor=BOX_COLOR
        )
        ax_img.add_patch(rect)

    ax_crops.axis("off")
    for i, p in enumerate(preds):
        r, c = divmod(i, cols)
        inset = ax_crops.inset_axes([
            c / cols + 0.01,
            1 - (r + 1) / rows + 0.06,
            1 / cols - 0.02,
            1 / rows - 0.1,
        ])
        inset.imshow(p["crop_pil"])
        inset.axis("off")
        inset.set_title(f"'{p['pred_label']}'", fontsize=CROP_LABEL_FONTSIZE)

    if ax_desc:
        ax_desc.axis("off")
        ax_desc.text(
            0.01, 0.5, description_text,
            fontsize=DESC_FONTSIZE,
            va="center", ha="left", wrap=True
        )

    plt.tight_layout()
    return fig


# ============================================================
# Main
# ============================================================

def main():
    enable_memory_growth()

    model = keras.models.load_model(MODEL_PATH)
    class_names = load_class_names(CLASS_NAMES_PATH)

    df = pd.read_csv(SPLIT_MANIFEST_PATH)
    df_test = df[df["new_split"] == "test"].copy()

    chosen_image_path = Path(random.choice(df_test["image_path"].unique()))
    df_img = df_test[df_test["image_path"] == str(chosen_image_path)]

    full_img = Image.open(chosen_image_path).convert("RGB")
    W, H = full_img.size

    scene_caption = blip_caption(full_img)
    (OUTPUT_DIR / "scene_caption.txt").write_text(scene_caption + "\n", encoding="utf-8")

    preds, sign_descs, alt_sentences = [], [], []

    for _, row in df_img.iterrows():
        bbox = clamp_bbox(row.xmin, row.ymin, row.xmax, row.ymax, W, H)
        if not bbox:
            continue

        crop = full_img.crop(bbox)
        probs = model.predict(preprocess_crop_for_model(crop), verbose=0)[0]
        pred_id = int(np.argmax(probs))
        pred_conf = float(probs[pred_id])

        if pred_conf < CONF_THRESH:
            continue

        pred_label = class_names[pred_id]

        size_desc = infer_relative_size(bbox, W, H)
        pos_desc = infer_position(bbox, W)

        sign_descs.append(f"a {size_desc} '{pred_label}' sign on the {pos_desc} side")

        alt_labels = [
            f"'{class_names[i]}'"
            for i in np.argsort(probs)[::-1][1:TOPK_FOR_TEXT + 1]
        ]
        if alt_labels:
            alt_sentences.append(
                f"For the sign on the {pos_desc} side, alternative candidate labels include {', '.join(alt_labels)}."
            )

        preds.append({"bbox": bbox, "crop_pil": crop, "pred_label": pred_label})

    context_text = build_context_description(scene_caption, sign_descs, alt_sentences)
    (OUTPUT_DIR / "context_aware_description.txt").write_text(context_text + "\n", encoding="utf-8")

    # Create figures
    fig1 = make_figure_input(full_img)
    fig2 = make_figure_panel(full_img, preds, "Classification Result")
    fig3 = make_figure_panel(full_img, preds, "Final Results (Classification + Context-Aware Scene Description)", context_text)

    # Save figures
    p1 = OUTPUT_DIR / "01_input_full_image.png"
    p2 = OUTPUT_DIR / "02_classification_panel.png"
    p3 = OUTPUT_DIR / "03_classification_panel_with_description.png"

    save_figure(fig1, p1)
    save_figure(fig2, p2)
    save_figure(fig3, p3)

    # Open figures sequentially (close to proceed)
    open_png_in_windows_blocking(p1, "Selected Input Image")
    open_png_in_windows_blocking(p2, "Classification Result")
    open_png_in_windows_blocking(p3, "Final Results")

    print("[DONE] Demo completed.")
    print(f"[DONE] Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
