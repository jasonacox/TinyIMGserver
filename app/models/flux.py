
"""
FLUX model handler for TinyIMGserver.

This module provides image generation using the FLUX.1-schnell model.
It includes GPU detection, memory optimization, and returns base64 encoded images.
"""

import os
import gc
import base64
import io
import torch
from diffusers import FluxPipeline

def generate_flux_image(prompt: str, height: int = 768, width: int = 768, steps: int = 6, seed: int = None) -> tuple[str, int]:
    """
    Generate an image using FLUX.1-schnell model.
    
    Args:
        prompt: Text description of the image to generate
        height: Image height in pixels (default: 768)
        width: Image width in pixels (default: 768)
        steps: Number of inference steps (default: 6)
        seed: Random seed for reproducible results (optional, random if None)
        
    Returns:
        tuple[str, int]: (Base64 encoded PNG image data, actual seed used)
        
    Raises:
        AssertionError: If no CUDA device is available
        Exception: If image generation fails
    """
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    assert torch.cuda.is_available(), "No CUDA device visible."
    free_bytes = []
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        free_bytes.append((free, i))
    gpu_index = max(free_bytes)[1]
    device = f"cuda:{gpu_index}"
    major_cc = torch.cuda.get_device_capability(gpu_index)[0]
    dtype = torch.bfloat16 if major_cc >= 8 else torch.float16

    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=dtype
    )
    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()

    # Use provided seed or generate random one
    if seed is None:
        import random
        seed = random.randint(0, 2**32 - 1)
    
    generator = torch.Generator(device=device).manual_seed(seed)
    image = pipe(
        prompt,
        guidance_scale=0.0,
        num_inference_steps=steps,
        height=height, width=width,
        generator=generator,
    ).images[0]

    # Convert PIL image to base64
    img_buffer = io.BytesIO()
    image.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')

    # Cleanup
    del image, pipe
    gc.collect()
    torch.cuda.empty_cache()
    
    return img_base64, seed
