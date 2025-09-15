#! /usr/bin/env python
"""
Image generation with FLUX.1-schnell

Requirements:
pip install diffusers transformers accelerate safetensors

Usage:
python flux.py
"""

import os, gc, torch
from diffusers import FluxPipeline

# 1) Reduce fragmentation before anything else
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# 2) Pick the GPU with the most free memory
assert torch.cuda.is_available(), "No CUDA device visible."
free_bytes = []
for i in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(i)
    free_bytes.append((free, i))
gpu_index = max(free_bytes)[1]
device = f"cuda:{gpu_index}"
print(f"Using device: {device}")

# 3) Choose safest dtype for your card
major_cc = torch.cuda.get_device_capability(gpu_index)[0]
dtype = torch.bfloat16 if major_cc >= 8 else torch.float16

# 4) Load once; apply memory savers
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=dtype
)
# Prefer offload over full .to(cuda) to avoid VRAM spike
pipe.enable_model_cpu_offload()
try:
    pipe.enable_attention_slicing()
except Exception:
    pass
try:
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
except Exception:
    pass
try:
    import xformers  # noqa: F401
    if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
        pipe.enable_xformers_memory_efficient_attention()
except Exception:
    pass

prompt = "A basket of kittens"

# 5) Conservative settings (adjust upward after it works)
height, width = 768, 768        # start smaller; try 896 or 1024 later
steps = 4                       # schnell looks ok at 4–10
guidance_scale = 1.0            # recommended for FLUX schnell

generator = torch.Generator(device=device).manual_seed(42)

def attempt(h, w, s, seq=False):
    if seq and hasattr(pipe, "enable_sequential_cpu_offload"):
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

cur_h, cur_w, cur_s = height, width, steps
attempted_seq = False
while True:
    try:
        image = attempt(cur_h, cur_w, cur_s, seq=attempted_seq)
        break
    except RuntimeError as re:
        msg = str(re)
        if ("CUDA out of memory" in msg) or ("CUDNN_STATUS_ALLOC_FAILED" in msg):
            print(f"OOM at {cur_w}x{cur_h}, steps={cur_s}. Retrying with safer settings...")
            torch.cuda.empty_cache()
            gc.collect()
            if not attempted_seq:
                attempted_seq = True
                continue
            if max(cur_h, cur_w) > 512:
                cur_h = max(384, cur_h - 128)
                cur_w = max(384, cur_w - 128)
                continue
            if cur_s > 4:
                cur_s = 4
                continue
        raise

image.save("basket-of-kittens.png")
print("Saved basket-of-kittens.png on", device)

# 6) Clean up if running repeatedly in one process
del image, pipe
gc.collect()
torch.cuda.empty_cache()