from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


def detect_resources(path: Path | None = None) -> dict[str, Any]:
    disk_path = path or Path.cwd()
    disk = shutil.disk_usage(disk_path)
    info: dict[str, Any] = {
        "cpu_count": os.cpu_count() or 1,
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
        "ffmpeg": bool(shutil.which("ffmpeg")),
        "ffprobe": bool(shutil.which("ffprobe")),
        "gpus": [],
    }
    if shutil.which("nvidia-smi"):
        try:
            proc = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader,nounits"],
                text=True,
                capture_output=True,
                check=True,
            )
            for line in proc.stdout.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    info["gpus"].append({"index": parts[0], "name": parts[1], "memory_total_mb": parts[2], "memory_free_mb": parts[3]})
        except Exception as exc:
            info["gpu_error"] = repr(exc)
    return info

