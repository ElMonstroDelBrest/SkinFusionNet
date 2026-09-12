"""Locate pip-installed NVIDIA CUDA/cuDNN libraries before loading ONNX Runtime."""
from __future__ import annotations

import os
import pathlib

_CONFIGURED = False


def configure_nvidia_libs() -> list[str]:
    global _CONFIGURED
    dirs: list[str] = []
    try:
        import torch

        site = pathlib.Path(torch.__file__).resolve().parents[1]
        dirs.extend(str(path) for path in site.glob("nvidia/*/lib") if path.is_dir())
    except Exception:
        pass
    try:
        import nvidia

        root = pathlib.Path(nvidia.__file__).resolve().parent
        dirs.extend(str(path) for path in root.glob("*/lib") if path.is_dir())
    except Exception:
        pass
    unique = list(dict.fromkeys(dirs))
    if unique:
        current = os.environ.get("LD_LIBRARY_PATH", "")
        prefix = os.pathsep.join(unique)
        if prefix not in current.split(os.pathsep):
            os.environ["LD_LIBRARY_PATH"] = (
                prefix if not current else prefix + os.pathsep + current
            )
    _CONFIGURED = True
    return unique
