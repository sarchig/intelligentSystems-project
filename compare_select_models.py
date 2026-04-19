"""
compare_select_models.py

Each model directory must contain:
- best_model.keras
- class_names.json
- split_manifest.csv

This script:
1) Finds trained model directories under outputs/train_runs/
2) Evaluates each model on:
   - validation split (new_split == "val")
   - test split       (new_split == "test")
3) Visualizes and saves:
   - metric comparison table
   - CM grid (one per model)
   - Option 1: normalized CM (per model)
   - Option 2: top-K CM (per model)
   - Option 3: log-scaled CM (per model)
   - Option 4: per-class accuracy bar (per model)
4) Selects and exports best models:
   - outputs/train_runs/best_performance_from_validation/
   - outputs/train_runs/best_performance_from_test/
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


# ==========================
# User Settings
# ==========================

RUNS_ROOT = Path("outputs/train_runs_Scenario3")
MODEL_FILE_NAME = "best_model.keras"

# Keep as-is (your previous setting)
BATCH_SIZE_EVAL = 64
IMG_SIZE = (224, 224)

SHOW_PLOTS = True
SAVE_PLOTS = True
MAX_MODELS_TO_DISPLAY = 12

TOPK_CM = 30
TOPK_CLASS_ACC = 50

SELECTION_WEIGHTS = {
    "loss": 1.0,
    "accuracy": 1.0,
    "precision": 1.0,
    "recall": 1.0,
    "f1": 1.0,
}

BEST_VAL_DIR = RUNS_ROOT / "best_performance_from_validation"
BEST_TEST_DIR = RUNS_ROOT / "best_performance_from_test"


# ==========================
# TensorFlow stability (recommended)
# ==========================

def enable_memory_growth() -> None:
    """Enable TF GPU memory growth to reduce OOM risk during evaluation."""
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
    except Exception as e:
        print("[WARN] Could not set memory growth:", e)


# ==========================
# Utilities
# ==========================

def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _reset_dir(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def _load_image_tf(path: tf.Tensor) -> tf.Tensor:
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def _make_dataset_from_df(df: pd.DataFrame, label_to_idx: Dict[str, int]) -> tf.data.Dataset:
    paths = df["crop_path"].astype(str).tolist()
    labels = [label_to_idx[str(l)] for l in df["label"].astype(str).tolist()]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda p, y: (_load_image_tf(p), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE_EVAL).prefetch(tf.data.AUTOTUNE)
    return ds


def discover_runs(runs_root: Path) -> List[Path]:
    """
    Discover model run directories under outputs/train_runs.
    Excludes:
      - best_performance_from_*
      - Previous Results (or any folder not containing required files)
    """
    run_dirs: List[Path] = []
    if not runs_root.exists():
        raise FileNotFoundError(f"RUNS_ROOT not found: {runs_root}")

    for p in sorted(runs_root.iterdir()):
        if not p.is_dir():
            continue

        # Skip export directories
        if p.name.startswith("best_performance_from"):
            continue

        # Skip your archived results folder
        if p.name.lower().startswith("previous"):
            continue

        # Only accept directories that contain required files
        if (p / MODEL_FILE_NAME).exists() and (p / "class_names.json").exists() and (p / "split_manifest.csv").exists():
            run_dirs.append(p)

    return run_dirs


def load_run_metadata(run_dir: Path) -> Tuple[pd.DataFrame, List[str], Dict[str, int]]:
    df = pd.read_csv(run_dir / "split_manifest.csv")
    class_names = json.loads((run_dir / "class_names.json").read_text(encoding="utf-8"))
    label_to_idx = {name: i for i, name in enumerate(class_names)}
    return df, class_names, label_to_idx


def evaluate_one_split(run_dir: Path, split_name: str) -> Dict[str, object]:
    model = keras.models.load_model(run_dir / MODEL_FILE_NAME)

    df, class_names, label_to_idx = load_run_metadata(run_dir)

    df_split = df[df["new_split"].astype(str) == split_name].copy()
    df_split = df_split[df_split["crop_path"].map(lambda p: Path(p).exists())].copy()

    if len(df_split) == 0:
        raise ValueError(f"[{run_dir.name}] No samples found for split={split_name}")

    ds = _make_dataset_from_df(df_split, label_to_idx)

    # Keras evaluate returns [loss, accuracy] because you compiled metrics=["accuracy"]
    eval_out = model.evaluate(ds, verbose=0)
    loss = float(eval_out[0])
    acc = float(eval_out[1]) if len(eval_out) > 1 else float("nan")

    y_true_all: List[np.ndarray] = []
    y_pred_all: List[np.ndarray] = []

    for x, y in ds:
        probs = model.predict(x, verbose=0)
        pred = np.argmax(probs, axis=1)
        y_true_all.append(y.numpy())
        y_pred_all.append(pred)

    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    support = np.bincount(y_true, minlength=len(class_names))

    return {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "split": split_name,
        "num_classes": len(class_names),
        "num_samples": int(len(df_split)),
        "loss": loss,
        "accuracy": acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "cm": cm,
        "support": support,
        "class_names": class_names,
    }


def rank_select_best(df_metrics: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    df = df_metrics.copy()

    df["r_loss"] = df["loss"].rank(method="min", ascending=True)
    df["r_accuracy"] = df["accuracy"].rank(method="min", ascending=False)
    df["r_precision"] = df["precision"].rank(method="min", ascending=False)
    df["r_recall"] = df["recall"].rank(method="min", ascending=False)
    df["r_f1"] = df["f1"].rank(method="min", ascending=False)

    df["rank_score"] = (
        weights["loss"] * df["r_loss"]
        + weights["accuracy"] * df["r_accuracy"]
        + weights["precision"] * df["r_precision"]
        + weights["recall"] * df["r_recall"]
        + weights["f1"] * df["r_f1"]
    )

    df = df.sort_values("rank_score", ascending=True).reset_index(drop=True)
    return df


# ==========================
# Plotting helpers
# ==========================

def plot_metrics_table(df_metrics: pd.DataFrame, title: str, save_path: Path = None) -> None:
    df_show = df_metrics.copy()
    cols = ["run_name", "loss", "accuracy", "precision", "recall", "f1", "num_samples"]
    df_show = df_show[cols].copy()

    for c in ["loss", "accuracy", "precision", "recall", "f1"]:
        df_show[c] = df_show[c].map(lambda v: f"{float(v):.4f}")

    fig = plt.figure(figsize=(14, 0.6 + 0.35 * len(df_show)))
    ax = plt.gca()
    ax.axis("off")
    ax.set_title(title, fontsize=14, pad=12)

    table = ax.table(
        cellText=df_show.values,
        colLabels=df_show.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    plt.tight_layout()

    if SAVE_PLOTS and save_path is not None:
        fig.savefig(save_path, dpi=250)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def plot_confusion_matrices_grid(results: List[Dict[str, object]], title: str, save_path: Path = None) -> None:
    n = min(len(results), MAX_MODELS_TO_DISPLAY)
    results = results[:n]

    cols = 2 if n <= 2 else 3
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 5.0 * rows))
    axes = np.array(axes).reshape(-1)

    for i in range(rows * cols):
        ax = axes[i]
        if i >= n:
            ax.axis("off")
            continue

        r = results[i]
        cm = r["cm"]
        run_name = r["run_name"]

        ax.imshow(cm, interpolation="nearest")
        ax.set_title(run_name, fontsize=10)
        ax.set_xlabel("Pred")
        ax.set_ylabel("True")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if SAVE_PLOTS and save_path is not None:
        fig.savefig(save_path, dpi=250)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def plot_cm_normalized(cm: np.ndarray, title: str, save_path: Path = None) -> None:
    cm = cm.astype(np.float32)
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sum, out=np.zeros_like(cm), where=(row_sum != 0))

    fig = plt.figure(figsize=(7, 6))
    plt.imshow(cm_norm, vmin=0.0, vmax=1.0, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    if SAVE_PLOTS and save_path is not None:
        fig.savefig(save_path, dpi=250)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def plot_cm_topk(cm: np.ndarray, support: np.ndarray, class_names: List[str], k: int, title: str, save_path: Path = None) -> None:
    top_idx = np.argsort(support)[::-1][:k]
    top_idx = np.sort(top_idx)

    cm_top = cm[np.ix_(top_idx, top_idx)]
    top_names = [class_names[i] for i in top_idx]

    fig = plt.figure(figsize=(10, 9))
    plt.imshow(cm_top, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    plt.xlabel("Predicted (Top-K)")
    plt.ylabel("True (Top-K)")
    plt.xticks(range(k), top_names, rotation=90, fontsize=6)
    plt.yticks(range(k), top_names, fontsize=6)
    plt.tight_layout()

    if SAVE_PLOTS and save_path is not None:
        fig.savefig(save_path, dpi=300)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def plot_cm_logscaled(cm: np.ndarray, title: str, save_path: Path = None) -> None:
    fig = plt.figure(figsize=(7, 6))
    plt.imshow(np.log1p(cm), interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    if SAVE_PLOTS and save_path is not None:
        fig.savefig(save_path, dpi=250)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def plot_per_class_accuracy(cm: np.ndarray, support: np.ndarray, class_names: List[str], k: int, title: str, save_path: Path = None) -> None:
    diag = np.diag(cm).astype(np.float32)
    denom = cm.sum(axis=1).astype(np.float32)
    acc_per_class = np.divide(diag, denom, out=np.zeros_like(diag), where=(denom != 0))

    top_idx = np.argsort(support)[::-1][:k]
    top_acc = acc_per_class[top_idx]
    top_names = [class_names[i] for i in top_idx]

    fig = plt.figure(figsize=(12, 4))
    plt.bar(range(k), top_acc)
    plt.title(title)
    plt.xlabel("Top-K classes by support")
    plt.ylabel("Per-class accuracy")
    plt.ylim(0.0, 1.0)
    plt.xticks(range(k), top_names, rotation=90, fontsize=6)
    plt.tight_layout()

    if SAVE_PLOTS and save_path is not None:
        fig.savefig(save_path, dpi=300)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def generate_all_cm_visuals_for_result(r: Dict[str, object], out_dir: Path) -> List[Path]:
    run_name = r["run_name"]
    split_name = r["split"]
    cm = r["cm"]
    support = r["support"]
    class_names = r["class_names"]

    saved: List[Path] = []

    p1 = out_dir / f"{split_name}_cm_normalized_{run_name}.png"
    plot_cm_normalized(cm, f"[{split_name}] Normalized CM - {run_name}", save_path=p1)
    saved.append(p1)

    p2 = out_dir / f"{split_name}_cm_top{TOPK_CM}_{run_name}.png"
    plot_cm_topk(cm, support, class_names, TOPK_CM, f"[{split_name}] Top-{TOPK_CM} CM - {run_name}", save_path=p2)
    saved.append(p2)

    p3 = out_dir / f"{split_name}_cm_log_{run_name}.png"
    plot_cm_logscaled(cm, f"[{split_name}] Log-scaled CM - {run_name}", save_path=p3)
    saved.append(p3)

    p4 = out_dir / f"{split_name}_per_class_acc_top{TOPK_CLASS_ACC}_{run_name}.png"
    plot_per_class_accuracy(cm, support, class_names, TOPK_CLASS_ACC, f"[{split_name}] Per-class Acc (Top-{TOPK_CLASS_ACC}) - {run_name}", save_path=p4)
    saved.append(p4)

    return saved


def export_best_run(best_dir: Path, best_row: pd.Series, split_tag: str, df_ranked: pd.DataFrame,
                    global_assets: List[Path], per_model_assets: List[Path]) -> None:
    _reset_dir(best_dir)

    src_run = Path(best_row["run_dir"])
    best_run_name = str(best_row["run_name"])

    shutil.copy2(src_run / MODEL_FILE_NAME, best_dir / MODEL_FILE_NAME)
    shutil.copy2(src_run / "class_names.json", best_dir / "class_names.json")
    shutil.copy2(src_run / "split_manifest.csv", best_dir / "split_manifest.csv")

    (best_dir / "best_run.txt").write_text(
        f"Selected best run based on {split_tag} metrics:\n{src_run}\n",
        encoding="utf-8",
    )

    metrics_out = {
        "split_basis": split_tag,
        "run_name": str(best_row["run_name"]),
        "run_dir": str(best_row["run_dir"]),
        "loss": float(best_row["loss"]),
        "accuracy": float(best_row["accuracy"]),
        "precision": float(best_row["precision"]),
        "recall": float(best_row["recall"]),
        "f1": float(best_row["f1"]),
        "rank_score": float(best_row["rank_score"]),
        "num_samples": int(best_row["num_samples"]),
    }
    (best_dir / f"best_{split_tag}_metrics.json").write_text(
        json.dumps(metrics_out, indent=2),
        encoding="utf-8",
    )

    df_ranked.to_csv(best_dir / f"ranked_{split_tag}_metrics.csv", index=False)

    for p in global_assets:
        if p.exists():
            shutil.copy2(p, best_dir / p.name)

    for p in per_model_assets:
        if p.exists() and best_run_name in p.name:
            shutil.copy2(p, best_dir / p.name)


def main() -> None:
    enable_memory_growth()

    run_dirs = discover_runs(RUNS_ROOT)
    if len(run_dirs) == 0:
        raise RuntimeError(f"No valid run directories found under: {RUNS_ROOT}")

    print(f"[INFO] Found {len(run_dirs)} runs.")
    for p in run_dirs:
        print(f"  - {p.name}")

    _safe_mkdir(RUNS_ROOT)

    # =========================
    # (1) Validation comparison
    # =========================
    val_results: List[Dict[str, object]] = []
    for rd in run_dirs:
        val_results.append(evaluate_one_split(rd, "val"))

    df_val = pd.DataFrame([{
        "run_dir": r["run_dir"],
        "run_name": r["run_name"],
        "loss": r["loss"],
        "accuracy": r["accuracy"],
        "precision": r["precision"],
        "recall": r["recall"],
        "f1": r["f1"],
        "num_samples": r["num_samples"],
    } for r in val_results])

    df_val_ranked = rank_select_best(df_val, SELECTION_WEIGHTS)

    val_table_path = RUNS_ROOT / "validation_metrics_table.png"
    val_cm_grid_path = RUNS_ROOT / "validation_confusion_matrices_grid.png"

    plot_metrics_table(df_val_ranked, "Validation Metrics Comparison (best_model.keras)", save_path=val_table_path)

    name_to_res = {r["run_name"]: r for r in val_results}
    val_results_ranked = [name_to_res[n] for n in df_val_ranked["run_name"].tolist() if n in name_to_res]
    plot_confusion_matrices_grid(val_results_ranked, "Validation Confusion Matrices (best first)", save_path=val_cm_grid_path)

    val_assets_all: List[Path] = []
    for r in val_results:
        val_assets_all.extend(generate_all_cm_visuals_for_result(r, RUNS_ROOT))

    best_val_row = df_val_ranked.iloc[0]
    export_best_run(
        best_dir=BEST_VAL_DIR,
        best_row=best_val_row,
        split_tag="validation",
        df_ranked=df_val_ranked,
        global_assets=[val_table_path, val_cm_grid_path],
        per_model_assets=val_assets_all,
    )
    print(f"[DONE] Best-by-validation exported to: {BEST_VAL_DIR}")

    # ===================
    # (3) Test comparison
    # ===================
    test_results: List[Dict[str, object]] = []
    for rd in run_dirs:
        test_results.append(evaluate_one_split(rd, "test"))

    df_test = pd.DataFrame([{
        "run_dir": r["run_dir"],
        "run_name": r["run_name"],
        "loss": r["loss"],
        "accuracy": r["accuracy"],
        "precision": r["precision"],
        "recall": r["recall"],
        "f1": r["f1"],
        "num_samples": r["num_samples"],
    } for r in test_results])

    df_test_ranked = rank_select_best(df_test, SELECTION_WEIGHTS)

    test_table_path = RUNS_ROOT / "test_metrics_table.png"
    test_cm_grid_path = RUNS_ROOT / "test_confusion_matrices_grid.png"

    plot_metrics_table(df_test_ranked, "Test Metrics Comparison (best_model.keras)", save_path=test_table_path)

    name_to_res_t = {r["run_name"]: r for r in test_results}
    test_results_ranked = [name_to_res_t[n] for n in df_test_ranked["run_name"].tolist() if n in name_to_res_t]
    plot_confusion_matrices_grid(test_results_ranked, "Test Confusion Matrices (best first)", save_path=test_cm_grid_path)

    test_assets_all: List[Path] = []
    for r in test_results:
        test_assets_all.extend(generate_all_cm_visuals_for_result(r, RUNS_ROOT))

    best_test_row = df_test_ranked.iloc[0]
    export_best_run(
        best_dir=BEST_TEST_DIR,
        best_row=best_test_row,
        split_tag="test",
        df_ranked=df_test_ranked,
        global_assets=[test_table_path, test_cm_grid_path],
        per_model_assets=test_assets_all,
    )
    print(f"[DONE] Best-by-test exported to: {BEST_TEST_DIR}")

    print("[ALL DONE] Comparison + selection completed.")


if __name__ == "__main__":
    main()