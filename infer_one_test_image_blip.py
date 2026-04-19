from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
import tkinter as tk

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np
import pandas as pd
from PIL import Image, ImageTk
from screeninfo import get_monitors

import tensorflow as tf
from tensorflow import keras


# ============================================================
# Configuration
# ============================================================

BEST_MODEL_DIR = Path("outputs/train_runs_Scenario2/best_performance_from_validation")

MODEL_PATH = BEST_MODEL_DIR / "best_model.keras"
CLASS_NAMES_PATH = BEST_MODEL_DIR / "class_names.json"
SPLIT_MANIFEST_PATH = BEST_MODEL_DIR / "split_manifest.csv"

OUTPUT_DIR = BEST_MODEL_DIR / "demo_one_random_test_image"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (224, 224)
CONF_THRESH = 0.0

NEAR_THRESH = 0.08
FAR_THRESH = 0.02

TOPK_FOR_TEXT = 1
CROP_GRID_COLS = 4

BOX_COLOR = "red"
BOX_LINEWIDTH = 2.5
BOX_FILL = False

TITLE_FONTSIZE = 20
CROP_LABEL_FONTSIZE = 9
DESC_FONTSIZE = 14

BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"


# ============================================================
# TensorFlow GPU configuration
# ============================================================

def enable_memory_growth() -> None:
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
    except Exception as e:
        print("[WARN] Could not enable memory growth:", e)


# ============================================================
# Path helpers
# ============================================================

def normalize_path_str(p: str | Path) -> str:
    s = str(p).strip().replace("\\", "/")
    s = os.path.normpath(s).replace("\\", "/")
    s = s.lower()
    return s


# ============================================================
# Windows blocking image viewer using Tkinter
# ============================================================

def open_png_in_windows_blocking(png_path: Path, window_title: str = "Result") -> None:
    png_path = png_path.resolve()

    if not png_path.exists():
        print(f"[WARN] File not found: {png_path}")
        return

    root = tk.Tk()
    root.title(window_title)
    root.configure(bg="white")

    win_w = 1200
    win_h = 800

    pointer_x = root.winfo_pointerx()
    pointer_y = root.winfo_pointery()

    monitors = get_monitors()
    target_monitor = None
    for m in monitors:
        if m.x <= pointer_x < m.x + m.width and m.y <= pointer_y < m.y + m.height:
            target_monitor = m
            break

    if target_monitor is None:
        target_monitor = monitors[0]

    pos_x = target_monitor.x + (target_monitor.width - win_w) // 2
    pos_y = target_monitor.y + (target_monitor.height - win_h) // 2

    root.geometry(f"{win_w}x{win_h}+{pos_x}+{pos_y}")

    img = Image.open(png_path).convert("RGB")
    img.thumbnail((win_w - 40, win_h - 60), Image.Resampling.LANCZOS)

    tk_img = ImageTk.PhotoImage(img)

    label = tk.Label(root, image=tk_img, bg="white")
    label.image = tk_img
    label.pack(expand=True)

    root.mainloop()


def save_figure(fig, save_path: Path, dpi: int = 250) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {save_path}")


# ============================================================
# Helper functions
# ============================================================

def load_class_names(path: Path) -> list[str]:
    return json.loads(path.read_text(encoding="utf-8"))


def clamp_bbox(xmin, ymin, xmax, ymax, w, h):
    xmin = max(0, min(int(xmin), w - 1))
    ymin = max(0, min(int(ymin), h - 1))
    xmax = max(0, min(int(xmax), w - 1))
    ymax = max(0, min(int(ymax), h - 1))
    if xmax <= xmin or ymax <= ymin:
        return None
    return xmin, ymin, xmax, ymax


def preprocess_crop_for_model(crop_pil: Image.Image) -> np.ndarray:
    """
    Match the training input exactly.

    Training pipeline:
    - decode to RGB
    - tf.image.resize
    - cast to float32
    - no division by 255.0
    """
    crop = crop_pil.convert("RGB")
    arr = np.asarray(crop)
    arr = tf.convert_to_tensor(arr)
    arr = tf.image.resize(arr, IMG_SIZE)
    arr = tf.cast(arr, tf.float32)
    arr = arr.numpy()
    return np.expand_dims(arr, axis=0)


def infer_relative_size(bbox, img_w, img_h) -> str:
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
    x1, _, x2, _ = bbox
    cx = (x1 + x2) / 2.0
    if cx < img_w / 3:
        return "left"
    if cx < 2 * img_w / 3:
        return "center"
    return "right"


def blip_caption(full_image_pil: Image.Image) -> str:
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
# Plotting
# ============================================================

def make_figure_input(full_img: Image.Image) -> plt.Figure:
    fig = plt.figure(figsize=(10, 6))
    plt.imshow(full_img)
    plt.title("Selected Input Image", fontsize=TITLE_FONTSIZE)
    plt.axis("off")
    plt.tight_layout()
    return fig


def make_figure_panel(full_img: Image.Image, preds: list[dict], title: str, description_text: str | None = None) -> plt.Figure:
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

    if ax_desc is not None:
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
    df["image_path_norm"] = df["image_path"].astype(str).map(normalize_path_str)

    df_test = df[df["new_split"] == "test"].copy()

    chosen_image_raw = random.choice(df_test["image_path"].unique())
    chosen_image_path = Path(chosen_image_raw)
    chosen_image_norm = normalize_path_str(chosen_image_path)

    df_img = df_test[df_test["image_path_norm"] == chosen_image_norm].copy()

    if len(df_img) == 0:
        raise RuntimeError(f"No bbox rows found for image: {chosen_image_path}")

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

    fig1 = make_figure_input(full_img)
    fig2 = make_figure_panel(full_img, preds, "Classification Result")
    fig3 = make_figure_panel(
        full_img,
        preds,
        "Final Results (Classification + Context-Aware Scene Description)",
        context_text
    )

    p1 = OUTPUT_DIR / "01_input_full_image.png"
    p2 = OUTPUT_DIR / "02_classification_panel.png"
    p3 = OUTPUT_DIR / "03_classification_panel_with_description.png"

    save_figure(fig1, p1)
    save_figure(fig2, p2)
    save_figure(fig3, p3)

    open_png_in_windows_blocking(p1, "Selected Input Image")
    open_png_in_windows_blocking(p2, "Classification Result")
    open_png_in_windows_blocking(p3, "Final Results")

    print("[DONE] Demo completed.")
    print(f"[DONE] Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()