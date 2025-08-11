"""
GPU and system resource management for TinyIMGserver.

This module provides functionality to:
- Detect available GPUs (NVIDIA, Apple Silicon, CPU fallback)
- Manage GPU allocation with thread-safe locks
- Prevent concurrent image generation on the same GPU
- Handle resource timeouts and cleanup
"""

import subprocess
import platform
import os
import threading
import time
from app.core.config import CONFIG

def scan_gpus():
    """
    Scan system for available GPUs and return a list of dicts with GPU info.
    
    Supports NVIDIA (nvidia-smi), Apple Silicon (macOS), and fallback to CPU info.
    
    Returns:
        list: List of dictionaries containing GPU information with keys:
            - id: GPU identifier
            - memory: Available memory
            - name: GPU/CPU name
            - type: GPU type ("NVIDIA", "Apple", "CPU")
    """
    gpus = []
    system = platform.system()
    # Try NVIDIA GPUs first
    try:
        result = subprocess.run([
            "nvidia-smi", "--query-gpu=index,memory.total,name", "--format=csv,noheader"
        ], capture_output=True, text=True, check=True)
        for line in result.stdout.strip().split("\n"):
            idx, mem, name = [x.strip() for x in line.split(",")]
            gpus.append({
                "id": int(idx),
                "memory": mem,
                "name": name,
                "type": "NVIDIA"
            })
    except Exception:
        # If on macOS, check for Apple Silicon GPU
        if system == "Darwin" and platform.machine() == "arm64":
            try:
                mem = os.popen("sysctl hw.memsize").read().strip().split(": ")[-1]
                mem_gb = int(mem) // (1024 ** 3)
                gpus.append({
                    "id": 0,
                    "memory": f"{mem_gb}GB shared",
                    "name": "Apple Silicon GPU",
                    "type": "Apple"
                })
            except Exception:
                pass
        # Fallback: No GPU, show CPU info
        if not gpus:
            cpu = platform.processor() or platform.machine()
            gpus.append({
                "id": 0,
                "memory": "N/A",
                "name": f"CPU: {cpu}",
                "type": "CPU"
            })
    return gpus


# Global GPU resource tracker
GPUS = scan_gpus()

# Thread-safe GPU allocation manager
_gpu_locks = {gpu['id']: threading.Lock() for gpu in GPUS if gpu['type'] != 'CPU'}

def acquire_gpu(timeout: float = None):
    """
    Acquire a free GPU for image generation.
    
    This function attempts to acquire an exclusive lock on an available GPU.
    If no GPU is immediately available, it will wait up to the timeout period.
    
    Args:
        timeout: Maximum time to wait for a GPU (seconds). 
                If None, uses config default.
                
    Returns:
        int or None: GPU ID if acquired successfully, None if timeout
    """
    if timeout is None:
        timeout = CONFIG.get("image_generation_timeout", 60)
    start = time.time()
    while True:
        for gpu_id, lock in _gpu_locks.items():
            acquired = lock.acquire(blocking=False)
            if acquired:
                return gpu_id
        if time.time() - start > timeout:
            return None
        time.sleep(0.1)

def release_gpu(gpu_id):
    """
    Release the lock for the given GPU.
    
    Args:
        gpu_id: The GPU ID to release
    """
    if gpu_id in _gpu_locks:
        _gpu_locks[gpu_id].release()