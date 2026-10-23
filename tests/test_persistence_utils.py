"""
Unit tests for persistence_utils.py.
"""
import os
import tempfile
from pathlib import Path
import pytest
import numpy as np
from persistence_utils import atomic_json_save, safe_json_load, numpy_to_native


def test_numpy_to_native():
    assert isinstance(numpy_to_native(np.int64(42)), int)
    assert isinstance(numpy_to_native(np.float32(3.14)), float)
    assert isinstance(numpy_to_native(np.bool_(True)), bool)
    assert numpy_to_native(np.array([1, 2, 3])) == [1, 2, 3]


def test_atomic_json_save_and_safe_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        target_path = Path(tmpdir) / "test_data.json"
        payload = [
            {"id": 1, "score": np.float64(25.5), "active": np.bool_(True)},
            {"id": 2, "tags": np.array(["nba", "props"])},
        ]

        # Save
        success = atomic_json_save(target_path, payload)
        assert success is True
        assert target_path.exists()

        # Load
        loaded = safe_json_load(target_path)
        assert len(loaded) == 2
        assert loaded[0]["score"] == 25.5
        assert loaded[0]["active"] is True
        assert loaded[1]["tags"] == ["nba", "props"]

        # Ensure no temp files remained in tmpdir
        files_in_dir = list(Path(tmpdir).glob("*"))
        assert len(files_in_dir) == 1
        assert files_in_dir[0] == target_path


def test_atomic_overwrite():
    with tempfile.TemporaryDirectory() as tmpdir:
        target_path = Path(tmpdir) / "tracker.json"

        # First write
        atomic_json_save(target_path, [{"bet": "A"}])
        assert safe_json_load(target_path) == [{"bet": "A"}]

        # Overwrite
        atomic_json_save(target_path, [{"bet": "B"}, {"bet": "C"}])
        assert safe_json_load(target_path) == [{"bet": "B"}, {"bet": "C"}]


def test_safe_load_nonexistent():
    path = Path("non_existent_file_12345.json")
    assert safe_json_load(path, default=[]) == []
