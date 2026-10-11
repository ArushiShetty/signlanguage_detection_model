"""Datasets, callbacks, plots, and sequence augmentation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from src.config import ALPHABET_LABELS, SEQ_LEN


def make_image_dataset(
    directory: str | Path,
    image_size: int,
    batch_size: int = 64,
    class_names: list[str] | None = None,
    shuffle: bool = True,
    seed: int = 0,
) -> tf.data.Dataset:
    """Load RGB images as float32 in 0..255 (no extra normalization)."""
    names = class_names or ALPHABET_LABELS
    ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        class_names=names,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        color_mode="rgb",
    )
    return ds.prefetch(tf.data.AUTOTUNE)


def class_weights_if_imbalanced(y: np.ndarray, threshold: float = 1.5) -> dict[int, float] | None:
    """Return sklearn class_weight dict if max/min count ratio exceeds threshold."""
    counts = np.bincount(y)
    counts = counts[counts > 0]
    if len(counts) == 0:
        return None
    if counts.max() / max(counts.min(), 1) <= threshold:
        return None
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def default_callbacks(checkpoint_path: str | None, patience: int = 6, lr_patience: int = 3):
    cbs = [
        tf.keras.callbacks.EarlyStopping(
            patience=patience, restore_best_weights=True, monitor="val_accuracy"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5, patience=lr_patience, monitor="val_loss"
        ),
    ]
    if checkpoint_path:
        cbs.append(
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path, save_best_only=True, monitor="val_accuracy"
            )
        )
    return cbs


def plot_confusion_matrix(y_true, y_pred, labels: list[str], out_path: str | Path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, xticklabels=labels, yticklabels=labels, cmap="mako")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_history(history, out_path: str | Path):
    hist = history.history
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(hist.get("accuracy", []), label="train")
    axes[0].plot(hist.get("val_accuracy", []), label="val")
    axes[0].set_title("Accuracy")
    axes[0].legend()
    axes[1].plot(hist.get("loss", []), label="train")
    axes[1].plot(hist.get("val_loss", []), label="val")
    axes[1].set_title("Loss")
    axes[1].legend()
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def time_stretch_sequence(seq: np.ndarray, factor: float) -> np.ndarray:
    """Resample 30 frames to 0.8x-1.2x speed, then pad/crop to SEQ_LEN."""
    n = seq.shape[0]
    new_n = max(2, int(round(n / factor)))
    idx = np.linspace(0, n - 1, new_n)
    stretched = np.stack([seq[int(round(i))] for i in idx], axis=0)
    if stretched.shape[0] >= SEQ_LEN:
        return stretched[:SEQ_LEN]
    pad = SEQ_LEN - stretched.shape[0]
    return np.concatenate([stretched, np.repeat(stretched[-1:], pad, axis=0)], axis=0)


def augment_sequence(seq: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Time-stretch, Gaussian jitter, hand-feature scale, random frame dropout."""
    rng = rng or np.random.default_rng()
    out = seq.astype(np.float32).copy()
    factor = float(rng.uniform(0.8, 1.2))
    out = time_stretch_sequence(out, factor)
    out = out + rng.normal(0.0, 0.01, size=out.shape).astype(np.float32)
    scale = float(rng.uniform(0.9, 1.1))
    out[:, :126] *= scale
    n_drop = int(rng.integers(1, 4))
    drop_idx = rng.choice(SEQ_LEN, size=n_drop, replace=False)
    out[drop_idx] = 0.0
    return out.astype(np.float32)


def sequence_dataset(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    augment: bool = False,
    shuffle: bool = True,
) -> tf.data.Dataset:
    def _map(x, label):
        if augment:
            x = tf.numpy_function(
                lambda s: augment_sequence(s),
                [x],
                tf.float32,
            )
            x.set_shape((SEQ_LEN, X.shape[-1]))
        return x, label

    ds = tf.data.Dataset.from_tensor_slices((X.astype(np.float32), y.astype(np.int32)))
    if shuffle:
        ds = ds.shuffle(len(X))
    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
