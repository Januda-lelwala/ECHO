"""Centralized PyTorch device selection for model workloads."""

import logging

import torch

from .settings import settings


logger = logging.getLogger(__name__)


def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(
        backend is not None
        and backend.is_built()
        and backend.is_available()
    )


def get_torch_device() -> torch.device:
    """Resolve MODEL_DEVICE, preferring accelerators when set to auto."""
    requested = settings.MODEL_DEVICE.strip().lower()

    if requested == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        elif _mps_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    elif requested.startswith("cuda"):
        if not torch.cuda.is_available():
            logger.warning("MODEL_DEVICE=%s is unavailable; using CPU", requested)
            device = torch.device("cpu")
        else:
            device = torch.device(requested)
    elif requested == "mps":
        if not _mps_available():
            logger.warning("MODEL_DEVICE=mps is unavailable; using CPU")
            device = torch.device("cpu")
        else:
            device = torch.device("mps")
    elif requested == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError(
            "MODEL_DEVICE must be one of: auto, cpu, mps, cuda, cuda:<index>"
        )

    logger.info("Using PyTorch model device: %s", device)
    return device


def get_torch_dtype(device: torch.device) -> torch.dtype:
    """Use half precision on CUDA and stable float32 on CPU/Apple Metal."""
    return torch.float16 if device.type == "cuda" else torch.float32


def get_pipeline_device(device: torch.device):
    """Return a Transformers pipeline-compatible device value."""
    return -1 if device.type == "cpu" else device


def empty_device_cache(device: torch.device) -> None:
    """Release unused accelerator cache without assuming CUDA."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def get_device_status() -> dict[str, object]:
    """Return a small runtime status payload for diagnostics."""
    device = get_torch_device()
    return {
        "device": str(device),
        "accelerated": device.type in {"cuda", "mps"},
        "mps_built": bool(
            getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_built()
        ),
        "mps_available": _mps_available(),
        "dtype": str(get_torch_dtype(device)).removeprefix("torch."),
    }
