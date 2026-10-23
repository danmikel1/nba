"""
persistence_utils.py — Robust, Atomic File Persistence for NBA Intelligence Engine

Guarantees crash-proof and corruption-proof saving of JSON files (bet trackers,
prediction logs, parlay trackers, and caches) on Windows and Unix systems.
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


def numpy_to_native(obj: Any) -> Any:
    """Recursively convert NumPy data types to Python native types for JSON serialization."""
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def atomic_json_save(file_path: Path, data: Any, indent: int = 2) -> bool:
    """
    Atomically writes `data` to `file_path` as formatted JSON.

    Steps:
    1. Writes to a temporary file in the EXACT same parent directory (ensures same filesystem).
    2. Flushes buffers and syncs to disk (os.fsync) to ensure physical write.
    3. Replaces target file via atomic `os.replace`.
    4. Cleans up temp file if any error occurs.

    Returns:
        True on success, raises or returns False on failure.
    """
    path = Path(file_path).resolve()
    dir_name = path.parent
    dir_name.mkdir(parents=True, exist_ok=True)

    temp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False, encoding="utf-8") as tf:
            temp_path = Path(tf.name)
            json.dump(data, tf, indent=indent, ensure_ascii=False, default=numpy_to_native)
            tf.flush()
            os.fsync(tf.fileno())

        # Atomic replacement on Windows and POSIX
        os.replace(temp_path, path)
        return True
    except Exception as e:
        logger.error(f"Atomic JSON save failed for {path}: {e}")
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise e


def safe_json_load(file_path: Path, default: Any = None) -> Any:
    """
    Safely load a JSON file with UTF-8 encoding.
    Returns `default` if file does not exist or is empty.
    Raises JSONDecodeError if content is corrupted, preventing silent data loss.
    """
    path = Path(file_path)
    if not path.exists():
        return default

    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            return default
        return json.loads(content)


def load_ml_training_data(data_dir: Optional[Path] = None, filename_base: str = "ml_training_data"):
    """
    High-performance ML training data loader.
    Prioritizes .parquet (18x faster, 5x smaller) with transparent fallback to .csv.
    """
    import pandas as pd
    dd = Path(data_dir) if data_dir else Path("data")
    parquet_path = dd / f"{filename_base}.parquet"
    csv_path = dd / f"{filename_base}.csv"

    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path, engine="pyarrow")
        except Exception as e:
            logger.warning(f"Failed to load {parquet_path}, falling back to CSV: {e}")

    if csv_path.exists():
        return pd.read_csv(csv_path, low_memory=False)

    raise FileNotFoundError(f"Neither {parquet_path} nor {csv_path} found in {dd}")


def save_ml_training_data(df: Any, data_dir: Optional[Path] = None, filename_base: str = "ml_training_data"):
    """
    Save ML training data in both high-performance Parquet format and CSV format.
    """
    dd = Path(data_dir) if data_dir else Path("data")
    dd.mkdir(parents=True, exist_ok=True)
    parquet_path = dd / f"{filename_base}.parquet"
    csv_path = dd / f"{filename_base}.csv"

    try:
        df.to_parquet(parquet_path, engine="pyarrow", compression="snappy")
        logger.info(f"Saved Parquet training dataset to {parquet_path}")
    except Exception as e:
        logger.warning(f"Failed to save {parquet_path}: {e}")

    try:
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV training dataset to {csv_path}")
    except Exception as e:
        logger.warning(f"Failed to save {csv_path}: {e}")
