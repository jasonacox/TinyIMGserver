
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
from typing import Optional, Tuple
from diffusers import FluxPipeline

def generate_flux_image(
    prompt: str,
    height: int = 768,
    width: int = 768,
    steps: int = 6,
    seed: Optional[int] = None,
    guidance_scale: float = 1.0,
    gpu_id: Optional[int] = None,
) -> Tuple[str, int]:
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
    try:
        print(f"Starting FLUX image generation with prompt: {prompt}")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        assert torch.cuda.is_available(), "No CUDA device visible."
        # Respect assigned GPU if provided by resource manager
        if gpu_id is not None:
            gpu_index = int(gpu_id)
        else:
            free_bytes = []
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                free_bytes.append((free, i))
            gpu_index = max(free_bytes)[1]
        device = f"cuda:{gpu_index}"
        print(f"Using device: {device}")
        major_cc = torch.cuda.get_device_capability(gpu_index)[0]
        dtype = torch.bfloat16 if major_cc >= 8 else torch.float16

        print("Loading FLUX pipeline...")
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=dtype,
        )
        # Prefer offload over a full .to(device) to avoid big VRAM spikes
        pipe.enable_model_cpu_offload()
        # Memory savers
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
        try:
            # Some pipelines also support tiling; use if available
            if hasattr(pipe, "enable_vae_tiling"):
                pipe.enable_vae_tiling()
            if hasattr(pipe, "enable_vae_slicing"):
                pipe.enable_vae_slicing()
        except Exception:
            pass
        # Use memory efficient attention if xformers is present
        try:
            import xformers  # noqa: F401
            if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
                pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

        # Use provided seed or generate random one
        if seed is None:
            import random
            seed = random.randint(0, 2**32 - 1)
        
        print(f"Generating image with seed: {seed}, steps: {steps}, size: {width}x{height}")
        generator = torch.Generator(device=device).manual_seed(seed)

        # Helper to attempt generation with current settings
        def _attempt(h: int, w: int, s: int, use_seq_offload: bool = False):
            if use_seq_offload and hasattr(pipe, "enable_sequential_cpu_offload"):
                print("Enabling sequential CPU offload and retrying to reduce VRAM...")
                try:
                    pipe.enable_sequential_cpu_offload()
                except Exception:
                    pass
            return pipe(
                prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=s,
                height=h, width=w,
                generator=generator,
            ).images[0]

        # Try with graceful fallbacks on OOM
        attempted_seq = False
        cur_h, cur_w, cur_s = int(height), int(width), int(steps)
        while True:
            try:
                image = _attempt(cur_h, cur_w, cur_s, use_seq_offload=attempted_seq)
                break
            except RuntimeError as re:
                msg = str(re)
                if ("CUDA out of memory" in msg) or ("CUDNN_STATUS_ALLOC_FAILED" in msg):
                    print(f"OOM at {cur_w}x{cur_h}, steps={cur_s}. Trying to free cache and retry with safer settings...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    if not attempted_seq:
                        attempted_seq = True
                        continue
                    # Reduce size then steps progressively
                    if max(cur_h, cur_w) > 512:
                        cur_h = max(384, cur_h - 128)
                        cur_w = max(384, cur_w - 128)
                        continue
                    if cur_s > 4:
                        cur_s = 4
                        continue
                # Not an OOM, re-raise
                raise

        print("Converting image to base64...")
        # Convert PIL image to base64
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')

        print("Cleaning up...")
        # Cleanup
        del image, pipe
        gc.collect()
        torch.cuda.empty_cache()
        
        print("FLUX image generation completed successfully")
    return img_base64, seed
    except Exception as e:
        print(f"Error in FLUX image generation: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
