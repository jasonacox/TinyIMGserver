#!/usr/bin/env python3
"""
Simple FLUX model test and download.
"""

import sys

print("Testing FLUX model download...")

try:
    import torch
    print(f"✅ PyTorch available: {torch.__version__}")
    
    from diffusers import FluxPipeline
    print("✅ Diffusers FluxPipeline imported successfully")
    
    print("Downloading FLUX model (this may take several minutes)...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.float16
    )
    print("✅ FLUX model downloaded successfully!")
    
    # Test a simple generation
    print("Testing image generation...")
    image = pipe(
        "a simple test image",
        num_inference_steps=1,
        height=256,
        width=256
    ).images[0]
    print("✅ Image generation test successful!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

print("🎉 FLUX model is ready to use!")
