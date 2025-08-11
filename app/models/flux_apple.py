"""
FLUX model handler optimized for Apple Silicon.

This module provides image generation using the FLUX.1-schnell model
with Apple Silicon MPS acceleration.
"""

import os
import gc
import base64
import io
import torch
import random
from PIL import Image

def generate_flux_image(prompt: str, height: int = 768, width: int = 768, steps: int = 6, seed: int = None) -> tuple[str, int]:
    """
    Generate an image using FLUX.1-schnell model optimized for Apple Silicon.
    
    Args:
        prompt: Text description of the image to generate
        height: Image height in pixels (default: 768)
        width: Image width in pixels (default: 768)
        steps: Number of inference steps (default: 6)
        seed: Random seed for reproducible results (optional, random if None)
        
    Returns:
        tuple[str, int]: (Base64 encoded PNG image data, actual seed used)
    """
    # Set random seed if provided
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        # No manual seeding needed for MPS
        pass
    
    # Determine the best device for Apple Silicon
    if torch.backends.mps.is_available():
        device = "mps"
        print(f"Using Apple Silicon MPS acceleration")
    else:
        device = "cpu"
        print(f"MPS not available, using CPU")
    
    try:
        from diffusers import FluxPipeline
        
        # Get HF token from environment
        hf_token = os.getenv('HF_TOKEN')
        
        # Load pipeline with Apple Silicon optimizations
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.float16 if device == "mps" else torch.float32,
            device_map=None,  # Let PyTorch handle device placement
            token=hf_token  # Use environment token
        )
        
        # Move to device
        pipe = pipe.to(device)
        
        # Enable memory efficient attention if available
        try:
            pipe.enable_attention_slicing()
        except:
            pass
            
        # Generate image
        print(f"Generating {width}x{height} image with {steps} steps...")
        
        with torch.no_grad():
            image = pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                generator=torch.Generator(device=device).manual_seed(seed)
            ).images[0]
        
        # Clean up GPU memory
        del pipe
        if device == "mps":
            torch.mps.empty_cache()
        gc.collect()
        
        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        
        image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        print(f"✅ FLUX image generated successfully (seed: {seed})")
        return image_data, seed
        
    except Exception as e:
        # Clean up on error
        if device == "mps":
            torch.mps.empty_cache()
        gc.collect()
        print(f"❌ FLUX generation failed: {e}")
        raise e
