from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_FAILURE_EVENTS: tuple[dict[str, str], ...] = (
    {"id": "F1", "start": "2020-04-18 00:00:00", "end": "2020-04-18 23:59:00", "failure_type": "AirLeak"},
    {"id": "F2", "start": "2020-05-29 23:30:00", "end": "2020-05-30 06:00:00", "failure_type": "AirLeak"},
    {"id": "F3", "start": "2020-06-05 10:00:00", "end": "2020-06-07 14:30:00", "failure_type": "AirLeak"},
    {"id": "F4", "start": "2020-07-15 14:30:00", "end": "2020-07-15 19:00:00", "failure_type": "AirLeak"},
)


@dataclass
class DatasetSplit:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp | None
    val_end: pd.Timestamp | None
    test_start: pd.Timestamp
    test_end: pd.Timestamp | None


@dataclass
class PreprocessingArtifacts:
    run_id: str
    preprocessed_root: Path
    labeled_csv_path: Path
    split_dir: Path
    train_csv_path: Path
    val_csv_path: Path | None
    test_csv_path: Path
    metadata_path: Path


def load_metropt3(csv_path: str | Path, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Load MetroPT3 data with parsed timestamp and preserved original index column."""
    return pd.read_csv(csv_path, index_col=0, parse_dates=[timestamp_col], engine="python")


def add_failure_labels(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    failure_events: Iterable[dict[str, str]] = DEFAULT_FAILURE_EVENTS,
    label_col: str = "failure_label",
    failure_id_col: str = "failure_id",
    failure_type_col: str = "failure_type",
) -> pd.DataFrame:
    """Label rows that fall inside known failure intervals."""
    out = df.copy()
    out[label_col] = 0
    out[failure_id_col] = ""
    out[failure_type_col] = ""

    for event in failure_events:
        start = pd.to_datetime(event["start"])
        end = pd.to_datetime(event["end"])
        mask = (out[timestamp_col] >= start) & (out[timestamp_col] <= end)
        out.loc[mask, label_col] = 1
        out.loc[mask, failure_id_col] = event.get("id", "")
        out.loc[mask, failure_type_col] = event.get("failure_type", "")

    return out


def label_and_overwrite_csv(
    csv_path: str | Path,
    timestamp_col: str = "timestamp",
    failure_events: Iterable[dict[str, str]] = DEFAULT_FAILURE_EVENTS,
) -> pd.DataFrame:
    """Update original CSV in place by appending failure label columns."""
    csv_path = Path(csv_path)
    df = load_metropt3(csv_path, timestamp_col=timestamp_col)
    labeled = add_failure_labels(df, timestamp_col=timestamp_col, failure_events=failure_events)
    labeled.to_csv(csv_path)
    return labeled


def filter_by_date_range(
    df: pd.DataFrame,
    timestamp_col: str,
    data_start: str | pd.Timestamp | None = None,
    data_end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return rows within the inclusive date range when bounds are provided."""
    out = df
    if data_start is not None:
        out = out[out[timestamp_col] >= pd.to_datetime(data_start)]
    if data_end is not None:
        out = out[out[timestamp_col] <= pd.to_datetime(data_end)]
    return out.copy()


def split_by_explicit_ranges(
    df: pd.DataFrame,
    timestamp_col: str,
    train_start: str | pd.Timestamp,
    train_end: str | pd.Timestamp,
    val_start: str | pd.Timestamp | None,
    val_end: str | pd.Timestamp | None,
    test_start: str | pd.Timestamp,
    test_end: str | pd.Timestamp | None,
) -> DatasetSplit:
    """Split data by explicit train/val/test date ranges."""
    train_start_ts = pd.to_datetime(train_start)
    train_end_ts = pd.to_datetime(train_end)
    val_start_ts = pd.to_datetime(val_start) if val_start is not None else None
    val_end_ts = pd.to_datetime(val_end) if val_end is not None else None
    test_start_ts = pd.to_datetime(test_start)
    test_end_ts = pd.to_datetime(test_end) if test_end is not None else None

    train_mask = (df[timestamp_col] >= train_start_ts) & (df[timestamp_col] < train_end_ts)
    if val_start_ts is not None and val_end_ts is not None:
        val_mask = (df[timestamp_col] >= val_start_ts) & (df[timestamp_col] < val_end_ts)
    else:
        val_mask = pd.Series(False, index=df.index)
    if test_end_ts is None:
        test_mask = df[timestamp_col] >= test_start_ts
    else:
        test_mask = (df[timestamp_col] >= test_start_ts) & (df[timestamp_col] < test_end_ts)

    return DatasetSplit(
        train_df=df.loc[train_mask].copy(),
        val_df=df.loc[val_mask].copy(),
        test_df=df.loc[test_mask].copy(),
        train_start=train_start_ts,
        train_end=train_end_ts,
        val_start=val_start_ts,
        val_end=val_end_ts,
        test_start=test_start_ts,
        test_end=test_end_ts,
    )


def split_by_months(
    df: pd.DataFrame,
    timestamp_col: str,
    train_months: int,
    val_months: int | None,
    test_months: int | None,
) -> DatasetSplit:
    """Split data into consecutive train/val/test windows based on month offsets."""
    start = df[timestamp_col].min()
    train_start = start
    train_end = train_start + pd.DateOffset(months=train_months)
    if val_months is not None:
        val_start = train_end
        val_end = val_start + pd.DateOffset(months=val_months)
        test_start = val_end
    else:
        val_start = None
        val_end = None
        test_start = train_end
    test_end = test_start + pd.DateOffset(months=test_months) if test_months else None

    train_mask = (df[timestamp_col] >= train_start) & (df[timestamp_col] < train_end)
    if val_start is not None and val_end is not None:
        val_mask = (df[timestamp_col] >= val_start) & (df[timestamp_col] < val_end)
    else:
        val_mask = pd.Series(False, index=df.index)
    if test_end is None:
        test_mask = df[timestamp_col] >= test_start
    else:
        test_mask = (df[timestamp_col] >= test_start) & (df[timestamp_col] < test_end)

    return DatasetSplit(
        train_df=df.loc[train_mask].copy(),
        val_df=df.loc[val_mask].copy(),
        test_df=df.loc[test_mask].copy(),
        train_start=train_start,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        test_start=test_start,
        test_end=test_end,
    )


def save_preprocessed_outputs(
    df_labeled: pd.DataFrame,
    split: DatasetSplit,
    preprocessed_root: str | Path,
    source_csv_path: str | Path,
    timestamp_col: str,
    train_start: str | pd.Timestamp,
    train_end: str | pd.Timestamp,
    val_start: str | pd.Timestamp | None,
    val_end: str | pd.Timestamp | None,
    test_start: str | pd.Timestamp,
    test_end: str | pd.Timestamp | None,
) -> PreprocessingArtifacts:
    """Save labeled data and split train/val/test CSVs under a clear folder structure."""
    preprocessed_root = Path(preprocessed_root)
    labeled_dir = preprocessed_root / "labeled"
    splits_dir = preprocessed_root / "splits"
    labeled_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    labeled_csv_path = labeled_dir / "MetroPT3_labeled.csv"
    df_labeled.to_csv(labeled_csv_path)

    run_id = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    split_dir = splits_dir / run_id
    split_dir.mkdir(parents=True, exist_ok=True)

    train_csv_path = split_dir / "train_split.csv"
    val_csv_path = split_dir / "val_split.csv" if val_start is not None and val_end is not None else None
    test_csv_path = split_dir / "test_split.csv"
    metadata_path = split_dir / "split_metadata.json"

    split.train_df.to_csv(train_csv_path)
    if val_csv_path is not None:
        split.val_df.to_csv(val_csv_path)
    split.test_df.to_csv(test_csv_path)

    metadata = {
        "run_id": run_id,
        "source_csv_path": str(source_csv_path),
        "timestamp_col": timestamp_col,
        "train_start": str(pd.to_datetime(train_start)),
        "train_end": str(pd.to_datetime(train_end)),
        "val_start": str(pd.to_datetime(val_start)) if val_start is not None else None,
        "val_end": str(pd.to_datetime(val_end)) if val_end is not None else None,
        "test_start": str(pd.to_datetime(test_start)),
        "test_end": str(pd.to_datetime(test_end)) if test_end is not None else None,
        "train_rows": int(len(split.train_df)),
        "val_rows": int(len(split.val_df)),
        "test_rows": int(len(split.test_df)),
        "saved_files": {
            "labeled_csv": str(labeled_csv_path),
            "train_csv": str(train_csv_path),
            "val_csv": str(val_csv_path) if val_csv_path is not None else None,
            "test_csv": str(test_csv_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return PreprocessingArtifacts(
        run_id=run_id,
        preprocessed_root=preprocessed_root,
        labeled_csv_path=labeled_csv_path,
        split_dir=split_dir,
        train_csv_path=train_csv_path,
        val_csv_path=val_csv_path,
        test_csv_path=test_csv_path,
        metadata_path=metadata_path,
    )


def label_split_and_save(
    csv_path: str | Path,
    preprocessed_root: str | Path,
    timestamp_col: str = "timestamp",
    failure_events: Iterable[dict[str, str]] = DEFAULT_FAILURE_EVENTS,
    train_start: str | pd.Timestamp | None = None,
    train_end: str | pd.Timestamp | None = None,
    val_start: str | pd.Timestamp | None = None,
    val_end: str | pd.Timestamp | None = None,
    test_start: str | pd.Timestamp | None = None,
    test_end: str | pd.Timestamp | None = None,
    train_months: int | None = None,
    val_months: int | None = None,
    test_months: int | None = None,
) -> PreprocessingArtifacts:
    """Label original CSV and persist train/val/test datasets using explicit ranges."""
    labeled = label_and_overwrite_csv(
        csv_path,
        timestamp_col=timestamp_col,
        failure_events=failure_events,
    )

    if train_start is not None and train_end is not None and test_start is not None:
        split = split_by_explicit_ranges(
            labeled,
            timestamp_col=timestamp_col,
            train_start=train_start,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
            test_start=test_start,
            test_end=test_end,
        )
    elif train_months is not None and val_months is not None:
        split = split_by_months(
            labeled,
            timestamp_col=timestamp_col,
            train_months=train_months,
            val_months=val_months,
            test_months=test_months,
        )
    elif train_months is not None:
        split = split_by_months(
            labeled,
            timestamp_col=timestamp_col,
            train_months=train_months,
            val_months=None,
            test_months=test_months,
        )
    else:
        raise ValueError("Provide explicit train/val/test ranges or month offsets.")

    return save_preprocessed_outputs(
        df_labeled=labeled,
        split=split,
        preprocessed_root=preprocessed_root,
        source_csv_path=csv_path,
        timestamp_col=timestamp_col,
        train_start=split.train_start,
        train_end=split.train_end,
        val_start=split.val_start,
        val_end=split.val_end,
        test_start=split.test_start,
        test_end=split.test_end,
    )


def find_split_run_by_name(preprocessed_root: str | Path, split_run_name: str) -> Path:
    """Find a saved split run by its folder name."""
    run_dir = Path(preprocessed_root) / "splits" / split_run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"No split run found at {run_dir}")
    return run_dir


def default_feature_cols(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    exclude_cols: Iterable[str] = ("failure_label", "failure_id", "failure_type"),
) -> list[str]:
    exclude = set(exclude_cols)
    return [
        c
        for c in df.columns
        if c != timestamp_col and c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]


def missing_nonmissing_summary(df: pd.DataFrame, feature_cols: Iterable[str]) -> pd.DataFrame:
    rows = []
    n = len(df)
    for c in feature_cols:
        missing = int(df[c].isna().sum())
        rows.append(
            {
                "feature": c,
                "missing_count": missing,
                "non_missing_count": n - missing,
                "missing_pct": round((missing / n * 100), 6) if n else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("feature")


def numeric_distribution_summary(df: pd.DataFrame, feature_cols: Iterable[str]) -> pd.DataFrame:
    """Return min/p10/median/p90/mean/std/max for selected features."""
    cols = list(feature_cols)
    if len(cols) == 0:
        return pd.DataFrame()

    summary = (
        df[cols]
        .describe(percentiles=[0.1, 0.5, 0.9])
        .T[["min", "10%", "50%", "90%", "mean", "std", "max"]]
        .rename(columns={"10%": "p10", "50%": "median", "90%": "p90"})
    )
    return summary.sort_index()
