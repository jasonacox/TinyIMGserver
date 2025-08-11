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
# If you still OOM without this, move it above .to(device) to load straight to CPU first.
pipe.enable_model_cpu_offload()        # big win for OOM (trades speed for RAM)
pipe.enable_attention_slicing()        # chunk attention to save memory
pipe.enable_vae_slicing()              # decode in slices (helps big images)
# pipe.enable_sequential_cpu_offload() # heavier offload; try if you're *still* OOM

prompt = "A basket of kittens"

# 5) Conservative settings (adjust upward after it works)
height, width = 768, 768        # start smaller; try 896 or 1024 later
steps = 6                       # schnell looks ok at 4–10
generator = torch.Generator(device=device).manual_seed(42)

image = pipe(
    prompt,
    guidance_scale=0.0,         # recommended for FLUX schnell
    num_inference_steps=steps,
    height=height, width=width,
    generator=generator,
).images[0]

image.save("basket-of-kittens.png")
print("Saved basket-of-kittens.png on", device)

# 6) Clean up if running repeatedly in one process
del image, pipe
gc.collect()
torch.cuda.empty_cache()