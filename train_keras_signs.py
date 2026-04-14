#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train a Keras image classifier using MTSD sign crops.

Key design:
- Crops are reshuffled and re-split into NEW train/val/test.
- This script trains using train split and evaluates ONLY on val split.
- Test split is saved for later use but NOT evaluated here.

IMPORTANT UPDATE (Selection A):
- Perform CLASS-WISE stratified split (8:1:1) so that each class is present in train/val/test.
- If a class has fewer than 10 samples, allow duplication (sampling with replacement)
  to ensure at least 1 sample in val and 1 sample in test, and the rest in train.
"""
import glob

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix, classification_report


# ============================================================
# Configuration (UNCHANGED params)
# ============================================================

CSV_PATH = Path("data/processed/mtsd_sign_crops_no_other_sign/crops.csv")
OUTPUT_ROOT = Path("outputs/train_runs")


IMG_SIZE = (224, 224)
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
SEED = 42

# New split ratios (must sum to 1.0)
TRAIN_RATIO = 0.80
VAL_RATIO = 0.1
TEST_RATIO = 0.1

MODEL_NAME = "ResNet50V2"  # change ONE LINE to compare models
# Options: "MobileNetV2", "ResNet152V2", "VGG16", "DenseNet201", "InceptionV3", "ResNet50V2"

# Early stopping (standard practice)
EARLY_STOP_PATIENCE = 3


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int) -> None:
    tf.keras.utils.set_random_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def make_run_dir(root: Path, model_name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = root / f"{ts}_{model_name}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_crops_csv(path: Path) -> pd.DataFrame:
    """
    crops.csv contains at least: crop_path, label (other-sign already excluded)
    """
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    if "crop_path" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain at least 'crop_path' and 'label' columns.")

    df = df[df["crop_path"].map(lambda p: Path(str(p)).exists())].copy()
    df["crop_path"] = df["crop_path"].astype(str)
    df["label"] = df["label"].astype(str)

    return df


def stratified_classwise_split_with_duplication(
    df: pd.DataFrame,
    train_r: float,
    val_r: float,
    test_r: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    CLASS-WISE split:
    - For each class, split into train/val/test according to 8:1:1.
    - If class has < 10 samples, sample WITH replacement so that:
        val >= 1, test >= 1, train >= 1 (if possible),
      while preserving approximate ratio.

    Implementation details:
    - For each label group:
        n = count
        desired_val  = max(1, round(n * val_r))
        desired_test = max(1, round(n * test_r))
        desired_train = max(1, n - desired_val - desired_test)
      If desired_train becomes 0, force it to 1 and adjust others minimally.
    - If n < desired_train + desired_val + desired_test, we oversample with replacement
      to reach total_needed.
    - Splits are produced by shuffling indices and slicing.
    """
    assert abs(train_r + val_r + test_r - 1.0) < 1e-6

    rng = np.random.default_rng(seed)

    train_parts = []
    val_parts = []
    test_parts = []

    # Group by class label
    for label, g in df.groupby("label"):
        g = g.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = len(g)

        # Compute target counts per class
        val_n = int(round(n * val_r))
        test_n = int(round(n * test_r))

        # Enforce at least 1 val and 1 test per class (as requested)
        val_n = max(1, val_n)
        test_n = max(1, test_n)

        # Train gets the rest; ensure at least 1
        train_n = n - val_n - test_n
        if train_n < 1:
            train_n = 1

        total_needed = train_n + val_n + test_n

        # If not enough samples, oversample with replacement
        if n < total_needed:
            pick_idx = rng.integers(low=0, high=n, size=total_needed)
            g_expanded = g.iloc[pick_idx].copy().reset_index(drop=True)
        else:
            # If enough, just use shuffled group
            g_expanded = g.copy()

            # If we have more than needed (possible due to rounding), we can truncate
            if len(g_expanded) > total_needed:
                g_expanded = g_expanded.iloc[:total_needed].copy().reset_index(drop=True)

        # Split
        g_train = g_expanded.iloc[:train_n].copy()
        g_val = g_expanded.iloc[train_n:train_n + val_n].copy()
        g_test = g_expanded.iloc[train_n + val_n:train_n + val_n + test_n].copy()

        train_parts.append(g_train)
        val_parts.append(g_val)
        test_parts.append(g_test)

    df_train = pd.concat(train_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df_val = pd.concat(val_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df_test = pd.concat(test_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return df_train, df_val, df_test


def build_label_map(df_train: pd.DataFrame) -> Tuple[Dict[str, int], List[str]]:
    labels = sorted(df_train["label"].unique())
    label_to_idx = {l: i for i, l in enumerate(labels)}
    return label_to_idx, labels


def load_image(path: tf.Tensor) -> tf.Tensor:
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def make_dataset(df: pd.DataFrame, label_map: Dict[str, int], training: bool) -> tf.data.Dataset:
    paths = df["crop_path"].astype(str).tolist()
    labels = [label_map[l] for l in df["label"].astype(str)]

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED)

    ds = ds.map(lambda p, y: (load_image(p), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return ds


def get_backbone(model_name: str) -> keras.Model:
    if model_name == "MobileNetV2":
        return keras.applications.MobileNetV2(
            include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3)
        )
    if model_name == "ResNet152V2":
        return keras.applications.ResNet152V2(
            include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3)
        )
    if model_name == "VGG16":
        return keras.applications.VGG16(
            include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3)
        )
    if model_name == "DenseNet201":
        return keras.applications.DenseNet201(
            include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3)
        )
    if model_name == "InceptionV3":
        return keras.applications.InceptionV3(
            include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3)
        )
    if model_name == "ResNet50V2":
        return keras.applications.ResNet50V2(
            include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3)
        )
    raise ValueError(model_name)


def build_model(model_name: str, num_classes: int) -> keras.Model:
    backbone = get_backbone(model_name)
    backbone.trainable = False

    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = backbone(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    #x = layers.Dropout(0.2)(x)
    #x = layers.Dense(2048, activation = 'relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_training_curves(history: keras.callbacks.History, run_dir: Path) -> None:
    hist = history.history

    plt.figure()
    if "loss" in hist:
        plt.plot(hist["loss"], label="train_loss")
    if "val_loss" in hist:
        plt.plot(hist["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "loss_curve.png", dpi=200)
    plt.close()

    plt.figure()
    if "accuracy" in hist:
        plt.plot(hist["accuracy"], label="train_acc")
    if "val_accuracy" in hist:
        plt.plot(hist["val_accuracy"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "accuracy_curve.png", dpi=200)
    plt.close()


def save_confusion_matrix_image(cm: np.ndarray, run_dir: Path) -> None:
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (Val)")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(run_dir / "confusion_matrix.png", dpi=250)
    plt.close()


# ============================================================
# Main
# ============================================================

def main() -> None:
    set_seed(SEED)

    run_dir = make_run_dir(OUTPUT_ROOT, MODEL_NAME)

    df = load_crops_csv(CSV_PATH)

    #print(f"DF length: {len(df)}")

    # ✅ UPDATED: class-wise split with duplication for small classes
    df_train, df_val, df_test = stratified_classwise_split_with_duplication(
        df,
        TRAIN_RATIO,
        VAL_RATIO,
        TEST_RATIO,
        seed=SEED,
    )

    # Save split manifest (VERY IMPORTANT for later test usage)
    split_manifest = pd.concat([
        df_train.assign(new_split="train"),
        df_val.assign(new_split="val"),
        df_test.assign(new_split="test"),
    ])
    split_manifest.to_csv(run_dir / "split_manifest.csv", index=False)

    label_map, class_names = build_label_map(df_train)
    with open(run_dir / "class_names.json", "w") as f:
        json.dump(class_names, f, indent=2)

    train_ds = make_dataset(df_train, label_map, training=True)
    val_ds = make_dataset(df_val, label_map, training=False)


    model = build_model(MODEL_NAME, num_classes=len(class_names))

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(run_dir / "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=1,
            min_lr=1e-6,
            verbose=1
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # Save training curve figures (loss/accuracy)
    save_training_curves(history, run_dir)

    # Validation evaluation
    y_true, y_pred = [], []
    for x, y in val_ds:
        p = model.predict(x, verbose=0)
        y_true.append(y.numpy())
        y_pred.append(np.argmax(p, axis=1))

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    np.save(run_dir / "confusion_matrix.npy", cm)

    # Save confusion matrix as an image
    save_confusion_matrix_image(cm, run_dir)

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
    (run_dir / "classification_report.txt").write_text(report)

    model.save(run_dir / "final_model.keras")
    model.summary()

    print(f"[DONE] Training completed. Results saved to {run_dir}")


if __name__ == "__main__":
    main()
