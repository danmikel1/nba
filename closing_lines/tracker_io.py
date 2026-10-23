"""
closing_lines.tracker_io — Centralized, Atomic I/O for Closing Line Tools

Provides crash-proof, atomic reading and writing for bet_tracker.json,
along with shared CLV calculation logic.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

# Add root to sys.path if not present
import sys
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from persistence_utils import atomic_json_save, safe_json_load

logger = logging.getLogger(__name__)

# Resolves to <project_root>/data/bet_tracker.json
TRACKER_PATH = _ROOT / "data" / "bet_tracker.json"


def get_tracker_path() -> Path:
    """Return the absolute path to bet_tracker.json."""
    return TRACKER_PATH


def load_tracker(tracker_path: Optional[Path] = None) -> List[Dict]:
    """
    Safely load bet_tracker.json.
    Returns list of bet dicts. Raises RuntimeError on corrupt JSON to avoid silent data loss.
    """
    path = tracker_path or TRACKER_PATH
    if not path.exists():
        logger.warning(f"Tracker file not found at {path}")
        return []

    try:
        data = safe_json_load(path, default=[])
        return data if isinstance(data, list) else []
    except Exception as e:
        logger.error(f"Failed to read tracker file at {path}: {e}")
        raise RuntimeError(f"Corrupted or unreadable tracker file at {path}: {e}") from e


def save_tracker_atomic(data: List[Dict], tracker_path: Optional[Path] = None) -> bool:
    """
    Atomically save bet records to bet_tracker.json.
    """
    path = tracker_path or TRACKER_PATH
    try:
        return atomic_json_save(path, data, indent=2)
    except Exception as e:
        logger.error(f"Failed to atomically save tracker data to {path}: {e}")
        return False


def calculate_clv(opening_line: float, closing_line: float, side: str) -> float:
    """
    Calculate Closing Line Value (CLV) in points.
    CLV > 0 means the market moved in your favor.
    - OVER: closing_line - opening_line
    - UNDER: opening_line - closing_line
    """
    if pd.isna(opening_line) or pd.isna(closing_line):
        return 0.0
    delta = float(closing_line) - float(opening_line)
    if str(side).upper() in ["UNDER", "U"]:
        delta = -delta
    return round(delta, 2)
