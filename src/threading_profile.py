"""
Dynamic Threading Configuration and Resource Auto-Tuning for ALPR Pipeline.
Prevents thread contention between OpenCV, PyTorch, and ONNX Runtime on constrained environments (e.g. 2-vCPU Colab),
while maximizing throughput on multi-core workstations.
"""
import os
import logging
from typing import Dict, Any, Optional
import cv2
import torch

logger = logging.getLogger(__name__)

def configure_threading_profile(
    device: str = "cpu",
    config_threads: Optional[Dict[str, Any]] = None
) -> Dict[str, int]:
    """
    Configures cv2, torch, and returns recommended ORT thread settings.
    Rules:
    - If user specifies overrides in config or ALPR_NUM_THREADS env var, honor them.
    - If on GPU (CUDA) and host CPU <= 4 (e.g. Colab 2-vCPU):
        cv2=1, torch=1, ort=1. Eliminates CPU context switching and decode starvation.
    - If on GPU with >= 8 host CPUs:
        cv2=2, torch=2, ort=2.
    - If on CPU:
        cv2=max(1, min(4, total_cpus // 2)), torch=max(1, min(8, total_cpus)), ort=max(1, min(8, total_cpus)).
    """
    total_cpus = os.cpu_count() or 1
    cfg = config_threads or {}
    env_override = os.environ.get("ALPR_NUM_THREADS")

    if env_override and env_override.isdigit():
        forced = int(env_override)
        profile = {
            "cv2_threads": forced,
            "torch_threads": forced,
            "ort_intra_threads": forced,
            "ort_inter_threads": 1
        }
    elif cfg.get("profile") == "single_thread":
        profile = {"cv2_threads": 1, "torch_threads": 1, "ort_intra_threads": 1, "ort_inter_threads": 1}
    elif cfg.get("profile") == "max":
        profile = {
            "cv2_threads": total_cpus,
            "torch_threads": total_cpus,
            "ort_intra_threads": total_cpus,
            "ort_inter_threads": 1
        }
    elif device == "cuda":
        if total_cpus <= 4:
            # Constrained GPU environment (Colab 2 vCPUs)
            profile = {"cv2_threads": 1, "torch_threads": 1, "ort_intra_threads": 1, "ort_inter_threads": 1}
        else:
            profile = {"cv2_threads": 2, "torch_threads": 2, "ort_intra_threads": 2, "ort_inter_threads": 1}
    else:
        # Multi-core CPU environment
        cv2_t = max(1, min(4, total_cpus // 2))
        torch_t = max(1, min(8, total_cpus))
        profile = {
            "cv2_threads": cv2_t,
            "torch_threads": torch_t,
            "ort_intra_threads": torch_t,
            "ort_inter_threads": 1
        }

    # User-level fine-grained overrides
    if "cv2" in cfg:
        profile["cv2_threads"] = int(cfg["cv2"])
    if "torch" in cfg:
        profile["torch_threads"] = int(cfg["torch"])
    if "ort_intra" in cfg:
        profile["ort_intra_threads"] = int(cfg["ort_intra"])

    # Apply cv2 and torch settings
    cv2.setNumThreads(profile["cv2_threads"])
    try:
        torch.set_num_threads(profile["torch_threads"])
    except Exception as e:
        logger.debug(f"Could not set torch threads: {e}")

    logger.info(
        f"🧵 [Threading Profile] device={device} (CPUs={total_cpus}) -> "
        f"cv2={profile['cv2_threads']}, torch={profile['torch_threads']}, "
        f"ort_intra={profile['ort_intra_threads']}"
    )
    return profile
