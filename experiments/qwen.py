
#! /usr/bin/env python
"""
Image generation with Qwen-Image

Requirements:
pip install diffusers transformers accelerate safetensors

Usage:
python qwen.py
"""

import os, gc, torch
from diffusers import DiffusionPipeline

# 1) Reduce fragmentation before anything else
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

model_name = "Qwen/Qwen-Image"

# 2) Pick the GPU with the most free memory
if torch.cuda.is_available():
    free_bytes = []
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        free_bytes.append((free, i))
    gpu_index = max(free_bytes)[1]
    device = f"cuda:{gpu_index}"
    major_cc = torch.cuda.get_device_capability(gpu_index)[0]
    torch_dtype = torch.bfloat16 if major_cc >= 8 else torch.float16
else:
    torch_dtype = torch.float32
    device = "cpu"
print(f"Using device: {device} with dtype {torch_dtype}")

# 3) Load pipeline and apply memory savers
pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
pipe = pipe.to(device)
pipe.enable_model_cpu_offload()
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
pipe.enable_sequential_cpu_offload() # heavier offload; try if you're *still* OOM

prompt = "A basket full of kittens"
negative_prompt = " "

# 4) Conservative settings (adjust upward after it works)
height, width = 768, 768
steps = 20
generator = torch.Generator(device=device).manual_seed(42)

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=steps,
    true_cfg_scale=4.0,
    generator=generator
).images[0]

image.save("example.png")
print(f"Saved example.png on {device}")

# 5) Clean up if running repeatedly in one process
del image, pipe
gc.collect()
torch.cuda.empty_cache()