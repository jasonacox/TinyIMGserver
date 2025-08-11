"""
SDXL model handler optimized for Apple Silicon.

This module provides image generation using Stable Diffusion XL
with Apple Silicon MPS acceleration.
"""

import os
import gc
import base64
import io
import torch
import random

def generate_sdxl_image(prompt: str, height: int = 1024, width: int = 1024, steps: int = 20, seed: int = None) -> tuple[str, int]:
    """
    Generate an image using SDXL model optimized for Apple Silicon.
    
    Args:
        prompt: Text description of the image to generate
        height: Image height in pixels (default: 1024)
        width: Image width in pixels (default: 1024)
        steps: Number of inference steps (default: 20)
        seed: Random seed for reproducible results (optional, random if None)
        
    Returns:
        tuple[str, int]: (Base64 encoded PNG image data, actual seed used)
    """
    # Set random seed if provided
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    
    torch.manual_seed(seed)
    
    # Determine the best device for Apple Silicon
    if torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
        print("Using Apple Silicon MPS acceleration")
    else:
        device = "cpu"
        dtype = torch.float32
        print("MPS not available, using CPU")
    
    try:
        from diffusers import StableDiffusionXLPipeline
        
        # Get HF token from environment
        hf_token = os.getenv('HF_TOKEN')
        
        # Load pipeline with Apple Silicon optimizations
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=dtype,
            device_map=None,
            use_safetensors=True,
            token=hf_token  # Use environment token
        )
        
        # Move to device
        pipe = pipe.to(device)
        
        # Enable memory efficient attention
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
        
        print(f"✅ SDXL image generated successfully (seed: {seed})")
        return image_data, seed
        
    except Exception as e:
        # Clean up on error
        if device == "mps":
            torch.mps.empty_cache()
        gc.collect()
        print(f"❌ SDXL generation failed: {e}")
        raise e
