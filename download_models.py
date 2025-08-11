#!/usr/bin/env python3
"""
Model download script for TinyIMGserver.

This script downloads the required AI models to the local cache so they're
available when the server starts. This prevents long download times during
the first generation request.
"""

import os
import sys
import torch
from diffusers import FluxPipeline, StableDiffusionXLPipeline

def check_gpu():
    """Check if CUDA is available and return device info."""
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Models will run on CPU (very slow).")
        return "cpu", torch.float32
    
    gpu_count = torch.cuda.device_count()
    print(f"Found {gpu_count} GPU(s)")
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1024**3
        print(f"  GPU {i}: {props.name} ({memory_gb:.1f}GB)")
    
    # Use the first GPU and determine dtype
    major_cc = torch.cuda.get_device_capability(0)[0]
    dtype = torch.bfloat16 if major_cc >= 8 else torch.float16
    return "cuda:0", dtype

def download_flux():
    """Download FLUX.1-schnell model."""
    print("\n=== Downloading FLUX.1-schnell model ===")
    try:
        device, dtype = check_gpu()
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=dtype
        )
        print("✅ FLUX model downloaded successfully")
        return True
    except Exception as e:
        print(f"❌ Error downloading FLUX model: {e}")
        return False

def download_sdxl():
    """Download Stable Diffusion XL model."""
    print("\n=== Downloading Stable Diffusion XL model ===")
    try:
        device, dtype = check_gpu()
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=dtype
        )
        print("✅ SDXL model downloaded successfully")
        return True
    except Exception as e:
        print(f"❌ Error downloading SDXL model: {e}")
        return False

def main():
    """Main download function."""
    print("TinyIMGserver Model Downloader")
    print("==============================")
    
    # Check available disk space
    cache_dir = os.path.expanduser("~/.cache/huggingface")
    if os.path.exists(cache_dir):
        stat = os.statvfs(cache_dir)
        free_gb = (stat.f_bavail * stat.f_frsize) / 1024**3
        print(f"Available disk space: {free_gb:.1f}GB")
        if free_gb < 20:
            print("WARNING: Less than 20GB free space. Models may require 15-20GB total.")
    
    print("\nThis will download the following models:")
    print("- black-forest-labs/FLUX.1-schnell (~10GB)")
    print("- stabilityai/stable-diffusion-xl-base-1.0 (~7GB)")
    
    response = input("\nProceed with download? (y/N): ").strip().lower()
    if response != 'y':
        print("Download cancelled.")
        return
    
    # Download models
    flux_success = download_flux()
    sdxl_success = download_sdxl()
    
    print("\n=== Download Summary ===")
    if flux_success and sdxl_success:
        print("✅ All models downloaded successfully!")
        print("You can now start the TinyIMGserver.")
    else:
        print("❌ Some models failed to download.")
        print("Check the error messages above and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
