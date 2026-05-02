from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

import json
import numpy as np
import pandas as pd

# Hide noisy TensorFlow C++ warnings that can look like hard failures.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.get_logger().setLevel("ERROR")


@dataclass
class VAEConfig:
    latent_dim: int = 16
    architecture: str = "conv1d"  # conv1d | dense | lstm_autoencoder
    conv_filters: tuple[int, ...] = (32, 64)
    kernel_size: int = 3
    hidden_units: tuple[int, ...] = (64, 32)
    lstm_units: tuple[int, ...] = (64, 32)
    encoder_use_batchnorm: bool = False
    encoder_dropout_rate: float = 0.2


@dataclass
class TrainConfig:
    epochs: int = 15
    batch_size: int = 1024
    learning_rate: float = 1e-3
    beta: float = 1
    sample_from_mu: bool = True
    validation_split: float = 0
    gradient_clipnorm: float = 1.0
    random_seed: int = 42
    verbose_epoch: bool = True


@dataclass
class ThresholdConfig:
    method: str = "p-train"  # val_f1 | p-train | p-test | percentile | mean_std
    percentile: float = 97.5
    std_factor: float = 3.0


@keras.utils.register_keras_serializable()
class Sampling(layers.Layer):
    def call(self, inputs, training=False):
        mu, log_var = inputs
        eps = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(0.5 * log_var) * eps


def build_vae(window_size: int, n_features: int, cfg: VAEConfig):
    encoder_inputs = keras.Input(shape=(window_size, n_features), name="encoder_input")
    x = encoder_inputs

    def apply_encoder_hidden_block(inputs, units: int, layer_index: int, total_hidden_layers: int):
        y = layers.Dense(units, activation=None, name=f"enc_dense_{layer_index+1}")(inputs)
        if cfg.encoder_use_batchnorm:
            y = layers.BatchNormalization(name=f"enc_bn_{layer_index+1}")(y)
        y = layers.ReLU(name=f"enc_relu_{layer_index+1}")(y)
        if layer_index < total_hidden_layers - 1 and cfg.encoder_dropout_rate > 0:
            y = layers.Dropout(cfg.encoder_dropout_rate, name=f"enc_dropout_{layer_index+1}")(y)
        return y

    if cfg.architecture == "conv1d":
        for i, filters in enumerate(cfg.conv_filters):
            x = layers.Conv1D(
                filters=filters,
                kernel_size=cfg.kernel_size,
                padding="same",
                activation="relu",
                name=f"enc_conv1d_{i+1}",
            )(x)
        x = layers.Flatten(name="flatten_input")(x)
        for i, units in enumerate(cfg.hidden_units):
            x = apply_encoder_hidden_block(x, units, i, len(cfg.hidden_units))
    elif cfg.architecture == "lstm_autoencoder":
        # LSTM encoder: keep temporal structure and compress to a fixed vector.
        for i, units in enumerate(cfg.lstm_units[:-1]):
            x = layers.LSTM(units, return_sequences=True, name=f"enc_lstm_{i+1}")(x)
        x = layers.LSTM(cfg.lstm_units[-1], return_sequences=False, name=f"enc_lstm_{len(cfg.lstm_units)}")(x)
        for i, units in enumerate(cfg.hidden_units):
            x = apply_encoder_hidden_block(x, units, i, len(cfg.hidden_units))
    else:
        x = layers.Flatten(name="flatten_input")(x)
        for i, units in enumerate(cfg.hidden_units):
            x = apply_encoder_hidden_block(x, units, i, len(cfg.hidden_units))

    mu = layers.Dense(cfg.latent_dim, name="z_mean")(x)
    log_var = layers.Dense(cfg.latent_dim, name="z_log_var")(x)
    z = Sampling(name="z")([mu, log_var])
    encoder = keras.Model(encoder_inputs, [mu, log_var, z], name="encoder")

    latent_inputs = keras.Input(shape=(cfg.latent_dim,), name="decoder_input")
    y = latent_inputs
    for i, units in enumerate(reversed(cfg.hidden_units)):
        y = layers.Dense(units, activation="relu", name=f"dec_dense_{i+1}")(y)
    if cfg.architecture == "conv1d":
        y = layers.Dense(window_size * cfg.conv_filters[-1], activation="relu", name="dec_dense_to_seq")(y)
        y = layers.Reshape((window_size, cfg.conv_filters[-1]), name="dec_reshape_seq")(y)
        for i, filters in enumerate(reversed(cfg.conv_filters[:-1])):
            y = layers.Conv1D(
                filters=filters,
                kernel_size=cfg.kernel_size,
                padding="same",
                activation="relu",
                name=f"dec_conv1d_{i+1}",
            )(y)
        decoder_outputs = layers.Conv1D(
            filters=n_features,
            kernel_size=cfg.kernel_size,
            padding="same",
            activation="linear",
            name="decoder_output",
        )(y)
    elif cfg.architecture == "lstm_autoencoder":
        # LSTM decoder: expand latent to sequence, then reconstruct each time step.
        y = layers.RepeatVector(window_size, name="dec_repeat_vector")(y)
        for i, units in enumerate(reversed(cfg.lstm_units)):
            y = layers.LSTM(units, return_sequences=True, name=f"dec_lstm_{i+1}")(y)
        decoder_outputs = layers.TimeDistributed(
            layers.Dense(n_features, activation="linear"), name="decoder_output"
        )(y)
    else:
        y = layers.Dense(window_size * n_features, activation="linear", name="decoder_output_flat")(y)
        decoder_outputs = layers.Reshape((window_size, n_features), name="decoder_output")(y)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    return encoder, decoder


def _split_train_val(x: np.ndarray, val_ratio: float, seed: int):
    if val_ratio <= 0.0:
        return x, np.empty((0,) + x.shape[1:], dtype=x.dtype)
    n = len(x)
    n_val = max(1, int(n * val_ratio))
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return x[train_idx], x[val_idx]


def train_vae(
    train_windows: np.ndarray,
    vae_cfg: VAEConfig,
    train_cfg: TrainConfig,
    val_windows: np.ndarray | None = None,
):
    if len(train_windows) == 0:
        raise ValueError("No training windows provided. Generate windows first in engineer_feature.ipynb.")

    if not np.isfinite(train_windows).all():
        raise ValueError("Training windows contain NaN or inf values. Clean or re-generate windows.")
    if val_windows is not None and len(val_windows) and not np.isfinite(val_windows).all():
        raise ValueError("Validation windows contain NaN or inf values. Clean or re-generate windows.")

    tf.keras.utils.set_random_seed(train_cfg.random_seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    window_size = train_windows.shape[1]
    n_features = train_windows.shape[2]
    encoder, decoder = build_vae(window_size, n_features, vae_cfg)

    optimizer_kwargs = {"learning_rate": train_cfg.learning_rate}
    if train_cfg.gradient_clipnorm is not None and train_cfg.gradient_clipnorm > 0:
        optimizer_kwargs["clipnorm"] = float(train_cfg.gradient_clipnorm)
    optimizer = keras.optimizers.Adam(**optimizer_kwargs)

    if val_windows is not None and len(val_windows):
        x_train = train_windows
        x_val = val_windows
    else:
        x_train, x_val = _split_train_val(train_windows, train_cfg.validation_split, train_cfg.random_seed)

        if len(x_train) == 0:
            # If validation split consumed all data (tiny datasets), fall back to full train.
            x_train = train_windows
            x_val = np.empty((0,) + train_windows.shape[1:], dtype=train_windows.dtype)

    train_ds = tf.data.Dataset.from_tensor_slices(x_train)
    if len(x_train) > 1:
        train_ds = train_ds.shuffle(
            len(x_train),
            seed=train_cfg.random_seed,
            reshuffle_each_iteration=False,
        )
    train_ds = train_ds.batch(train_cfg.batch_size).prefetch(1)
    val_ds = tf.data.Dataset.from_tensor_slices(x_val).batch(train_cfg.batch_size) if len(x_val) else None

    history = {
        "epoch": [],
        "train_total_loss": [],
        "train_recon_loss": [],
        "train_kl_loss": [],
        "val_total_loss": [],
        "val_recon_loss": [],
        "val_kl_loss": [],
    }

    def train_step(batch_x):
        with tf.GradientTape() as tape:
            mu, log_var, z = encoder(batch_x, training=True)
            use_sampling = float(train_cfg.beta) > 0.0 or not train_cfg.sample_from_mu
            z_used = z if use_sampling else mu
            x_hat = decoder(z_used, training=True)

            recon = tf.reduce_mean(tf.math.squared_difference(batch_x, x_hat), axis=[1, 2])
            if float(train_cfg.beta) == 0.0:
                kl = tf.zeros_like(recon)
                total = tf.reduce_mean(recon)
            else:
                kl = -0.5 * tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=1)
                total = tf.reduce_mean(recon + train_cfg.beta * kl)

        vars_ = encoder.trainable_weights + decoder.trainable_weights
        grads = tape.gradient(total, vars_)
        optimizer.apply_gradients(zip(grads, vars_))
        return tf.reduce_mean(recon), tf.reduce_mean(kl), total

    def eval_step(batch_x):
        mu, log_var, z = encoder(batch_x, training=False)
        use_sampling = float(train_cfg.beta) > 0.0 or not train_cfg.sample_from_mu
        z_used = z if use_sampling else mu
        x_hat = decoder(z_used, training=False)
        recon = tf.reduce_mean(tf.math.squared_difference(batch_x, x_hat), axis=[1, 2])
        if float(train_cfg.beta) == 0.0:
            kl = tf.zeros_like(recon)
            total = tf.reduce_mean(recon)
        else:
            kl = -0.5 * tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=1)
            total = tf.reduce_mean(recon + train_cfg.beta * kl)
        return tf.reduce_mean(recon), tf.reduce_mean(kl), total

    for epoch in range(1, train_cfg.epochs + 1):
        tr_recon, tr_kl, tr_total, tr_n = 0.0, 0.0, 0.0, 0
        for batch in train_ds:
            r, k, t = train_step(batch)
            tr_recon += float(r)
            tr_kl += float(k)
            tr_total += float(t)
            tr_n += 1

        if val_ds is not None:
            va_recon, va_kl, va_total, va_n = 0.0, 0.0, 0.0, 0
            for batch in val_ds:
                r, k, t = eval_step(batch)
                va_recon += float(r)
                va_kl += float(k)
                va_total += float(t)
                va_n += 1
            va_recon /= max(1, va_n)
            va_kl /= max(1, va_n)
            va_total /= max(1, va_n)
        else:
            va_recon, va_kl, va_total = np.nan, np.nan, np.nan

        tr_recon /= max(1, tr_n)
        tr_kl /= max(1, tr_n)
        tr_total /= max(1, tr_n)

        history["epoch"].append(epoch)
        history["train_total_loss"].append(tr_total)
        history["train_recon_loss"].append(tr_recon)
        history["train_kl_loss"].append(tr_kl)
        history["val_total_loss"].append(va_total)
        history["val_recon_loss"].append(va_recon)
        history["val_kl_loss"].append(va_kl)

        if train_cfg.verbose_epoch:
            if np.isfinite(va_total):
                print(
                    f"Epoch {epoch}/{train_cfg.epochs} - "
                    f"train_total={tr_total:.6f}, train_recon={tr_recon:.6f}, train_kl={tr_kl:.6f}, "
                    f"val_total={va_total:.6f}, val_recon={va_recon:.6f}, val_kl={va_kl:.6f}"
                )
            else:
                print(
                    f"Epoch {epoch}/{train_cfg.epochs} - "
                    f"train_total={tr_total:.6f}, train_recon={tr_recon:.6f}, train_kl={tr_kl:.6f}"
                )

        if not np.isfinite(tr_total):
            raise ValueError(
                f"Training loss became non-finite at epoch {epoch}. "
                "Try lowering learning_rate, reducing model size, or increasing gradient clipping."
            )
        if val_ds is not None and not np.isfinite(va_total):
            raise ValueError(
                f"Validation loss became non-finite at epoch {epoch}. "
                "Check validation windows for invalid values and tune learning settings."
            )

    return encoder, decoder, history


def reconstruction_scores(encoder, decoder, windows: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """Backward-compatible score API: negative log reconstruction probability."""
    scores, _ = reconstruction_probability_scores(encoder, decoder, windows, batch_size=batch_size, sigma2=None)
    return scores


def mse_reconstruction_scores(encoder, decoder, windows: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """Return per-window reconstruction MSE (higher = more anomalous)."""
    ds = tf.data.Dataset.from_tensor_slices(windows).batch(batch_size)
    scores = []
    for batch_x in ds:
        mu, _, _ = encoder(batch_x, training=False)
        x_hat = decoder(mu, training=False)
        mse = tf.reduce_mean(tf.math.squared_difference(batch_x, x_hat), axis=[1, 2])
        scores.append(mse.numpy())
    return np.concatenate(scores, axis=0) if scores else np.empty((0,), dtype=np.float32)


def reconstruction_probability_scores(
    encoder,
    decoder,
    windows: np.ndarray,
    batch_size: int = 512,
    sigma2: float | None = None,
) -> tuple[np.ndarray, float]:
    """Return negative log reconstruction probability scores and the variance used.

    Lower reconstruction probability corresponds to higher NLL score.
    """
    ds = tf.data.Dataset.from_tensor_slices(windows).batch(batch_size)
    sse_all = []
    for batch_x in ds:
        mu, _, _ = encoder(batch_x, training=False)
        x_hat = decoder(mu, training=False)
        sse = tf.reduce_sum(tf.math.squared_difference(batch_x, x_hat), axis=[1, 2])
        sse_all.append(sse.numpy())

    if not sse_all:
        return np.empty((0,), dtype=np.float32), float("nan")

    sse_all = np.concatenate(sse_all, axis=0)
    d = windows.shape[1] * windows.shape[2]
    sigma2_used = float(np.var(sse_all / max(1, d)) + 1e-8) if sigma2 is None else float(max(sigma2, 1e-8))
    nll = 0.5 * (sse_all / sigma2_used + d * np.log(2.0 * np.pi * sigma2_used))
    return nll.astype(np.float32), sigma2_used


def compute_threshold(train_scores: np.ndarray, cfg: ThresholdConfig) -> float:
    if cfg.method == "mean_std":
        return float(np.mean(train_scores) + cfg.std_factor * np.std(train_scores))
    return float(np.percentile(train_scores, cfg.percentile))


def optimize_threshold_by_f1(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Find threshold maximizing F1 on a labeled validation set."""
    if len(scores) == 0:
        raise ValueError("Validation scores are empty.")

    labels = labels.astype(np.int32)
    if np.unique(labels).size < 2:
        # Degenerate labels; fall back to upper percentile threshold.
        th = float(np.percentile(scores, 95.0))
        return th, 0.0

    qs = np.linspace(0.01, 0.99, 200)
    thresholds = np.unique(np.quantile(scores, qs))

    best_f1 = -1.0
    best_th = float(thresholds[len(thresholds) // 2])
    for th in thresholds:
        pred = (scores > th).astype(np.int32)
        m = binary_metrics(labels, pred)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_th = float(th)
    return best_th, float(best_f1)


def select_threshold(
    train_scores: np.ndarray,
    cfg: ThresholdConfig,
    val_scores: np.ndarray | None = None,
    val_labels: np.ndarray | None = None,
    test_scores: np.ndarray | None = None,
) -> tuple[float, dict[str, float | str]]:
    """Select threshold according to configured strategy; supports val F1 maximization."""
    if cfg.method == "val_f1":
        if val_scores is None or val_labels is None:
            raise ValueError("val_scores and val_labels are required when method='val_f1'.")
        th, best_f1 = optimize_threshold_by_f1(val_scores, val_labels)
        return th, {"method": "val_f1", "best_val_f1": best_f1}

    if cfg.method == "p-test":
        if test_scores is None:
            raise ValueError("test_scores is required when method='p-test'.")
        th = float(np.percentile(test_scores, cfg.percentile))
        return th, {"method": "p-test", "percentile": cfg.percentile}

    if cfg.method in {"p-train", "percentile", "mean_std"}:
        th = compute_threshold(train_scores, cfg)
        method_name = "p-train" if cfg.method == "percentile" else cfg.method
        return th, {"method": method_name, "percentile": cfg.percentile}

    raise ValueError(f"Unknown threshold method: {cfg.method}")


def classify(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (scores > threshold).astype(np.int32)


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(np.int32)
    y_pred = y_pred.astype(np.int32)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max(1, len(y_true))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def roc_auc_binary(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute ROC-AUC from ranks; returns NaN if only one class is present."""
    y_true = y_true.astype(np.int32)
    scores = scores.astype(np.float64)

    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(np.sum(pos))
    n_neg = int(np.sum(neg))
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    sum_pos = float(np.sum(ranks[pos]))
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def find_latest_processed_window_run(processed_windows_root: str | Path, require_val: bool = False) -> Path:
    root = Path(processed_windows_root)
    runs = sorted([p for p in root.iterdir() if p.is_dir()])
    if require_val:
        runs = [p for p in runs if (p / "val_windows.npy").exists() and (p / "val_window_labels.npy").exists()]
    if not runs:
        if require_val:
            raise FileNotFoundError(
                f"No processed window runs with validation files found in {root}. "
                "Run preprocess_data.ipynb (2/1/1) then engineer_feature.ipynb."
            )
        raise FileNotFoundError(f"No processed window runs found in {root}")
    return runs[-1]


def find_processed_window_run_by_date(
    processed_windows_root: str | Path,
    target_date: str | pd.Timestamp,
    require_val: bool = False,
) -> Path:
    """Find a processed-window run whose saved date ranges cover target_date."""
    root = Path(processed_windows_root)
    target_ts = pd.to_datetime(target_date)
    runs = sorted([p for p in root.iterdir() if p.is_dir()])

    for run_dir in reversed(runs):
        meta_path = run_dir / "metadata.json"
        if not meta_path.exists():
            continue
        if require_val and not ((run_dir / "val_windows.npy").exists() and (run_dir / "val_window_labels.npy").exists()):
            continue

        metadata = json.loads(meta_path.read_text())
        for period_name in ("train_period", "val_period", "test_period"):
            period = metadata.get(period_name, {})
            start = period.get("start")
            end = period.get("end")
            if start is None or end is None:
                continue
            start_ts = pd.to_datetime(start)
            end_ts = pd.to_datetime(end)
            if start_ts <= target_ts <= end_ts:
                return run_dir

    raise FileNotFoundError(
        f"No processed window run found in {root} containing date {target_ts}."
    )


def find_processed_window_run_by_name(processed_windows_root: str | Path, run_name: str) -> Path:
    """Find a processed-window run by its folder name."""
    run_dir = Path(processed_windows_root) / run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"No processed window run found at {run_dir}")
    return run_dir


def load_window_run(run_dir: str | Path):
    run_dir = Path(run_dir)
    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata.json in {run_dir}")

    metadata = json.loads(meta_path.read_text())
    train_windows = np.load(run_dir / "train_windows.npy")
    val_windows = np.load(run_dir / "val_windows.npy") if (run_dir / "val_windows.npy").exists() else np.empty((0,), dtype=np.float32)
    test_windows = np.load(run_dir / "test_windows.npy")

    train_lbl_path = run_dir / "train_window_labels.npy"
    val_lbl_path = run_dir / "val_window_labels.npy"
    test_lbl_path = run_dir / "test_window_labels.npy"
    train_labels = np.load(train_lbl_path) if train_lbl_path.exists() else np.zeros((len(train_windows),), dtype=np.int32)
    val_labels = np.load(val_lbl_path) if val_lbl_path.exists() else np.zeros((len(val_windows),), dtype=np.int32)
    test_labels = np.load(test_lbl_path) if test_lbl_path.exists() else np.zeros((len(test_windows),), dtype=np.int32)

    # Engineer step may have flattened windows; reshape for model input.
    if train_windows.ndim == 2:
        w = metadata["params"]["window_size"]
        n_features = len(metadata["feature_cols"])
        train_windows = train_windows.reshape((-1, w, n_features))
        val_windows = val_windows.reshape((-1, w, n_features)) if len(val_windows) else val_windows
        test_windows = test_windows.reshape((-1, w, n_features))

    return (
        train_windows.astype(np.float32),
        val_windows.astype(np.float32),
        test_windows.astype(np.float32),
        train_labels.astype(np.int32),
        val_labels.astype(np.int32),
        test_labels.astype(np.int32),
        metadata,
    )


def save_training_artifacts(
    output_root: str | Path,
    encoder,
    decoder,
    history: dict,
    train_scores: np.ndarray,
    threshold: float,
    train_preds: np.ndarray,
    train_metrics: dict,
    vae_cfg: VAEConfig,
    train_cfg: TrainConfig,
    threshold_cfg: ThresholdConfig,
    source_run_dir: str | Path,
    project_root: str | Path | None = None,
):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    root = Path(project_root).resolve() if project_root is not None else output_root.parent.resolve()

    def _to_relative(path_value: str | Path) -> str:
        path_obj = Path(path_value)
        if not path_obj.is_absolute():
            return path_obj.as_posix()
        try:
            resolved = path_obj.resolve()
        except OSError:
            return str(path_value)
        try:
            return resolved.relative_to(root).as_posix()
        except ValueError:
            return str(path_value)

    run_id = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    encoder.save(run_dir / "encoder.keras")
    decoder.save(run_dir / "decoder.keras")

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    np.save(run_dir / "train_scores.npy", train_scores)
    np.save(run_dir / "train_predictions.npy", train_preds)

    summary = {
        "run_id": run_id,
        "source_data_run": _to_relative(source_run_dir),
        "vae_config": asdict(vae_cfg),
        "train_config": asdict(train_cfg),
        "threshold_config": asdict(threshold_cfg),
        "threshold": float(threshold),
        "train_samples": int(len(train_scores)),
        "train_metrics": train_metrics,
        "saved_files": {
            "encoder": _to_relative(run_dir / "encoder.keras"),
            "decoder": _to_relative(run_dir / "decoder.keras"),
            "history": _to_relative(run_dir / "history.csv"),
            "train_scores": _to_relative(run_dir / "train_scores.npy"),
            "train_predictions": _to_relative(run_dir / "train_predictions.npy"),
        },
    }

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return run_dir, summary


def plot_thresholds(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    title: str = "Reconstruction Error Distribution with Threshold",
):
    """
    Plot the distribution of reconstruction scores for normal and anomalous data.
    
    Parameters:
    -----------
    scores : np.ndarray
        Reconstruction scores (higher = more anomalous).
    labels : np.ndarray
        Binary labels (0 = normal, 1 = anomaly).
    threshold : float
        The threshold value to display on the plot.
    title : str
        Title of the plot.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
    
    # Separate normal and anomalous scores
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot histograms
    ax.hist(normal_scores, bins=50, alpha=0.6, label=f"Normal (n={len(normal_scores)})", color="blue", edgecolor="black")
    if len(anomaly_scores) > 0:
        ax.hist(anomaly_scores, bins=50, alpha=0.6, label=f"Anomaly (n={len(anomaly_scores)})", color="red", edgecolor="black")
    
    # Plot threshold line
    ax.axvline(threshold, color="green", linestyle="--", linewidth=2, label=f"Threshold = {threshold:.6f}")
    
    # Labels and legend
    ax.set_xlabel("Reconstruction Error (MSE)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


def load_ae_run(run_dir: str | Path, random_seed: int = 42):
    """
    Load a trained autoencoder run from a directory.

    Returns a dict with 'encoder', 'decoder', 'summary', and 'scores' keys.
    """
    run_dir = Path(run_dir)

    # Deterministic ops require a seed to be set before random ops execute.
    tf.keras.utils.set_random_seed(random_seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    
    # Load encoder and decoder
    encoder = keras.models.load_model(run_dir / "encoder.keras")
    decoder = keras.models.load_model(run_dir / "decoder.keras")
    
    # Load summary
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    
    # Load scores if available
    train_scores = np.load(run_dir / "train_scores.npy") if (run_dir / "train_scores.npy").exists() else None
    test_scores = np.load(run_dir / "test_scores.npy") if (run_dir / "test_scores.npy").exists() else None
    
    return {
        "encoder": encoder,
        "decoder": decoder,
        "summary": summary,
        "train_scores": train_scores,
        "test_scores": test_scores,
    }


def list_ae_runs(ae_root: str | Path) -> list[dict]:
    """
    List all available AE model runs with their metadata.
    
    Returns a list of dicts with 'run_name', 'run_path', and 'summary' keys.
    """
    ae_root = Path(ae_root)
    runs = []
    
    for run_dir in sorted(ae_root.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        
        summary_path = run_dir / "summary.json"
        summary = {}
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
        
        runs.append({
            "run_name": run_dir.name,
            "run_path": run_dir,
            "summary": summary,
        })
    
    return runs
