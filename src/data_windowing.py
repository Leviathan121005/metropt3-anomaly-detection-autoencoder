from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import json
import numpy as np
import pandas as pd

from data_preprocessing import default_feature_cols, load_metropt3


@dataclass
class FeatureEngineeringParams:
    train_csv_path: str
    val_csv_path: str | None
    test_csv_path: str
    timestamp_col: str = "timestamp"
    label_col: str = "failure_label"
    train_normal_only: bool = True
    feature_cols: tuple[str, ...] = ()
    window_size: int = 60
    stride: int = 10
    flatten_windows: bool = True
    window_label_strategy: str = "last_percent"  # positive_ratio | last_percent
    window_label_positive_ratio: float = 0.1
    window_label_last_percent: float = 10.0


@dataclass
class FeatureEngineeringArtifacts:
    run_dir: Path
    train_windows_path: Path
    val_windows_path: Path | None
    test_windows_path: Path
    train_window_labels_path: Path
    val_window_labels_path: Path | None
    test_window_labels_path: Path
    metadata_path: Path


def _write_windows_npy_from_starts(
    data: np.ndarray,
    out_path: Path,
    window_size: int,
    flatten: bool,
    starts: np.ndarray,
) -> tuple[int, ...]:
    n_rows, n_features = data.shape
    if n_rows < window_size or len(starts) == 0:
        shape = (0, window_size * n_features) if flatten else (0, window_size, n_features)
        mem = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=shape)
        del mem
        return shape

    shape = (len(starts), window_size * n_features) if flatten else (len(starts), window_size, n_features)
    mem = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=shape)

    for i, s in enumerate(starts):
        w = data[s : s + window_size]
        mem[i] = w.reshape(-1) if flatten else w

    mem.flush()
    del mem
    return shape


def _window_starts(n_rows: int, window_size: int, stride: int) -> np.ndarray:
    if n_rows < window_size:
        return np.empty((0,), dtype=np.int64)
    return np.arange(0, n_rows - window_size + 1, stride, dtype=np.int64)


def _window_label_from_slice(
    window_labels: np.ndarray,
    strategy: str,
    positive_ratio: float,
    last_percent: float,
) -> int:
    if strategy in {"last", "last_percent"}:
        # Label positive only if the last `last_percent` of the window are all positive
        tail_fraction = min(1.0, max(0.0, float(last_percent) / 100.0))
        tail_len = max(1, int(np.ceil(len(window_labels) * tail_fraction)))
        tail = window_labels[-tail_len:]
        return int(bool(np.all(tail == 1)))
    if strategy == "positive_ratio":
        return int(float(np.mean(window_labels == 1)) >= positive_ratio)
    return int(np.max(window_labels))


def _build_window_labels(
    labels: np.ndarray,
    window_size: int,
    stride: int,
    strategy: str = "positive_ratio",
    positive_ratio: float = 0.1,
    last_percent: float = 10.0,
) -> np.ndarray:
    # Not used in current pipeline; keep behavior via _build_window_labels_from_starts
    n_rows = len(labels)
    if n_rows < window_size:
        return np.empty((0,), dtype=np.int32)

    out: list[int] = []
    for s in range(0, n_rows - window_size + 1, stride):
        w = labels[s : s + window_size]
        out.append(_window_label_from_slice(w, strategy, positive_ratio, last_percent))
    return np.asarray(out, dtype=np.int32)


def _build_window_labels_from_starts(
    labels: np.ndarray,
    window_size: int,
    starts: np.ndarray,
    strategy: str = "positive_ratio",
    positive_ratio: float = 0.1,
    last_percent: float = 10.0,
) -> np.ndarray:
    if len(starts) == 0:
        return np.empty((0,), dtype=np.int32)
    out: list[int] = []
    for s in starts:
        w = labels[s : s + window_size]
        out.append(_window_label_from_slice(w, strategy, positive_ratio, last_percent))
    return np.asarray(out, dtype=np.int32)


def engineer_and_save_windows(
    params: FeatureEngineeringParams,
    output_root: str | Path,
) -> FeatureEngineeringArtifacts:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    use_validation = params.val_csv_path is not None

    train_df = load_metropt3(params.train_csv_path, timestamp_col=params.timestamp_col)
    val_df = load_metropt3(params.val_csv_path, timestamp_col=params.timestamp_col) if use_validation else pd.DataFrame(columns=train_df.columns)
    test_df = load_metropt3(params.test_csv_path, timestamp_col=params.timestamp_col)
    train_df = train_df.sort_values(params.timestamp_col).reset_index(drop=True)
    val_df = val_df.sort_values(params.timestamp_col).reset_index(drop=True)
    test_df = test_df.sort_values(params.timestamp_col).reset_index(drop=True)

    feature_source_df = train_df if len(train_df) else (val_df if len(val_df) else test_df)
    feature_cols = list(params.feature_cols) if params.feature_cols else default_feature_cols(feature_source_df, timestamp_col=params.timestamp_col)
    if len(feature_cols) != 15:
        feature_count_warning = f"Expected 15 features, got {len(feature_cols)}"
    else:
        feature_count_warning = ""

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    X_val = val_df[feature_cols].to_numpy(dtype=np.float32)
    X_test = test_df[feature_cols].to_numpy(dtype=np.float32)

    run_id = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    train_windows_path = run_dir / "train_windows.npy"
    val_windows_path = run_dir / "val_windows.npy" if use_validation else None
    test_windows_path = run_dir / "test_windows.npy"
    train_window_labels_path = run_dir / "train_window_labels.npy"
    val_window_labels_path = run_dir / "val_window_labels.npy" if use_validation else None
    test_window_labels_path = run_dir / "test_window_labels.npy"
    metadata_path = run_dir / "metadata.json"

    train_row_labels = train_df[params.label_col].to_numpy(dtype=np.int32) if params.label_col in train_df.columns else np.zeros((len(train_df),), dtype=np.int32)
    val_row_labels = val_df[params.label_col].to_numpy(dtype=np.int32) if params.label_col in val_df.columns else np.zeros((len(val_df),), dtype=np.int32)
    test_row_labels = test_df[params.label_col].to_numpy(dtype=np.int32) if params.label_col in test_df.columns else np.zeros((len(test_df),), dtype=np.int32)

    train_starts_all = _window_starts(len(train_df), params.window_size, params.stride)
    val_starts = _window_starts(len(val_df), params.window_size, params.stride)
    test_starts = _window_starts(len(test_df), params.window_size, params.stride)

    if params.train_normal_only and params.label_col in train_df.columns:
        train_starts = np.asarray(
            [
                int(s)
                for s in train_starts_all
                if np.max(train_row_labels[s : s + params.window_size]) == 0
            ],
            dtype=np.int64,
        )
    else:
        train_starts = train_starts_all

    train_windows_shape = _write_windows_npy_from_starts(
        X_train,
        train_windows_path,
        window_size=params.window_size,
        flatten=params.flatten_windows,
        starts=train_starts,
    )
    if use_validation and val_windows_path is not None:
        val_windows_shape = _write_windows_npy_from_starts(
            X_val,
            val_windows_path,
            window_size=params.window_size,
            flatten=params.flatten_windows,
            starts=val_starts,
        )
    else:
        val_windows_shape = None
    test_windows_shape = _write_windows_npy_from_starts(
        X_test,
        test_windows_path,
        window_size=params.window_size,
        flatten=params.flatten_windows,
        starts=test_starts,
    )

    train_window_labels = _build_window_labels_from_starts(
        train_row_labels,
        params.window_size,
        train_starts,
        strategy=params.window_label_strategy,
        positive_ratio=params.window_label_positive_ratio,
        last_percent=params.window_label_last_percent,
    )
    if use_validation:
        val_window_labels = _build_window_labels_from_starts(
            val_row_labels,
            params.window_size,
            val_starts,
            strategy=params.window_label_strategy,
            positive_ratio=params.window_label_positive_ratio,
            last_percent=params.window_label_last_percent,
        )
    else:
        val_window_labels = None
    test_window_labels = _build_window_labels_from_starts(
        test_row_labels,
        params.window_size,
        test_starts,
        strategy=params.window_label_strategy,
        positive_ratio=params.window_label_positive_ratio,
        last_percent=params.window_label_last_percent,
    )

    if params.train_normal_only and np.any(train_window_labels == 1):
        raise ValueError("train_normal_only=True but train windows still contain label 1.")

    np.save(train_window_labels_path, train_window_labels)
    if use_validation and val_window_labels_path is not None and val_window_labels is not None:
        np.save(val_window_labels_path, val_window_labels)
    np.save(test_window_labels_path, test_window_labels)

    metadata = {
        "run_id": run_id,
        "params": asdict(params),
        "feature_cols": feature_cols,
        "feature_count_warning": feature_count_warning,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)) if use_validation else 0,
        "test_rows": int(len(test_df)),
        "train_candidate_windows": int(len(train_starts_all)),
        "train_kept_windows": int(len(train_starts)),
        "train_windows_shape": list(train_windows_shape),
        "test_windows_shape": list(test_windows_shape),
        "train_period": {
            "start": str(train_df[params.timestamp_col].min()) if len(train_df) else None,
            "end": str(train_df[params.timestamp_col].max()) if len(train_df) else None,
        },
        "test_period": {
            "start": str(test_df[params.timestamp_col].min()) if len(test_df) else None,
            "end": str(test_df[params.timestamp_col].max()) if len(test_df) else None,
        },
        "saved_files": {
            "train_windows": str(train_windows_path),
            "test_windows": str(test_windows_path),
            "train_window_labels": str(train_window_labels_path),
            "test_window_labels": str(test_window_labels_path),
        },
    }

    if use_validation and val_windows_shape is not None:
        metadata["val_windows_shape"] = list(val_windows_shape)
        metadata["val_period"] = {
            "start": str(val_df[params.timestamp_col].min()) if len(val_df) else None,
            "end": str(val_df[params.timestamp_col].max()) if len(val_df) else None,
        }
        metadata["saved_files"]["val_windows"] = str(val_windows_path)
        metadata["saved_files"]["val_window_labels"] = str(val_window_labels_path)

    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return FeatureEngineeringArtifacts(
        run_dir=run_dir,
        train_windows_path=train_windows_path,
        val_windows_path=val_windows_path,
        test_windows_path=test_windows_path,
        train_window_labels_path=train_window_labels_path,
        val_window_labels_path=val_window_labels_path,
        test_window_labels_path=test_window_labels_path,
        metadata_path=metadata_path,
    )
