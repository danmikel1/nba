"""
hardware_utils.py — GPU Acceleration & Hardware Detection Utility

Automatically detects NVIDIA CUDA GPUs (like the RTX 5060 Ti 16GB) and configures
optimal runtime acceleration for XGBoost, PyTorch, and Monte Carlo simulation engines.
"""

import logging
from typing import Any, Dict, Optional
import xgboost as xgb

logger = logging.getLogger(__name__)

_DEVICE_CACHE: Optional[str] = None
_GPU_INFO_CACHE: Optional[Dict[str, Any]] = None


def detect_cuda_device() -> str:
    """
    Test and return 'cuda' if NVIDIA GPU acceleration is functional in XGBoost,
    otherwise fallback to 'cpu'. Cached after first check.
    """
    global _DEVICE_CACHE
    if _DEVICE_CACHE is not None:
        return _DEVICE_CACHE

    try:
        # Fast test with small dummy matrix
        import numpy as np
        X = np.random.randn(50, 4)
        y = np.random.randn(50)
        test_model = xgb.XGBRegressor(
            n_estimators=1,
            max_depth=1,
            tree_method="hist",
            device="cuda",
            verbosity=0,
        )
        test_model.fit(X, y)
        _DEVICE_CACHE = "cuda"
        logger.info("CUDA GPU acceleration verified active (device='cuda').")
    except Exception as e:
        logger.info(f"CUDA not usable ({e}); falling back to CPU (device='cpu').")
        _DEVICE_CACHE = "cpu"

    return _DEVICE_CACHE


def get_gpu_info() -> Dict[str, Any]:
    """
    Retrieve GPU hardware specifications (name, VRAM, driver).
    """
    global _GPU_INFO_CACHE
    if _GPU_INFO_CACHE is not None:
        return _GPU_INFO_CACHE

    info = {
        "has_gpu": False,
        "name": "CPU Only",
        "total_vram_gb": 0.0,
        "driver_version": "N/A",
        "device": detect_cuda_device(),
    }

    try:
        import subprocess
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if res.returncode == 0 and res.stdout.strip():
            parts = [p.strip() for p in res.stdout.strip().split(",")]
            if len(parts) >= 3:
                info["has_gpu"] = True
                info["name"] = parts[0]
                info["total_vram_gb"] = round(float(parts[1]) / 1024.0, 1)
                info["driver_version"] = parts[2]
    except Exception:
        pass

    _GPU_INFO_CACHE = info
    return info


def check_cuda_availability() -> tuple[bool, str]:
    """
    Returns (is_cuda_available, gpu_name_or_reason).
    """
    info = get_gpu_info()
    if info.get("has_gpu") and info.get("device") == "cuda":
        return True, info.get("name", "NVIDIA CUDA GPU")
    return False, "CPU"


def get_xgboost_hardware_params(base_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Returns base_params updated with optimal device and tree_method.
    """
    params = dict(base_params) if base_params else {}
    device = detect_cuda_device()

    if device == "cuda":
        params["device"] = "cuda"
        params["tree_method"] = "hist"
    else:
        params["device"] = "cpu"
        params["tree_method"] = "hist"

    return params
