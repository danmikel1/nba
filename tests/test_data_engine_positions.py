import json
from pathlib import Path
import pytest

from data_engine import DataEngine


def test_position_cache_loading(tmp_path):
    # prepare a fake player positions file in a temporary data directory
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    positions = {"12345": "PF", "67890": "SG"}
    with open(data_dir / "player_positions.json", "w") as f:
        json.dump({"time": "2026-01-01T00:00:00", "positions": positions}, f)
    
    # initialize engine with fetching disabled (so it only uses disk cache)
    engine = DataEngine(data_dir, season="2025-26", allow_fetch=False)
    # cached entries should be loaded into memory
    assert engine._position_cache.get(12345) == "PF"
    assert engine._position_cache.get(67890) == "SG"
    # lookups should return the correct value without defaulting to SF
    assert engine._get_real_position(12345) == "PF"
    assert engine._get_real_position(67890) == "SG"

    # non-present ID should fall back to SF but still be written back to disk
    val = engine._get_real_position(99999)
    assert val == "SF"
    with open(data_dir / "player_positions.json") as f:
        newdata = json.load(f)
    assert "99999" in newdata.get("positions", {})
