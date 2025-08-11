#! /usr/bin/env python
"""
Image generation with Stable Diffusion XL (SDXL)

Requirements:
pip install diffusers transformers accelerate safetensors

Usage:
python sdxl.py
"""

from diffusers import DiffusionPipeline
import torch

# 1) Check for CUDA
assert torch.cuda.is_available(), "No CUDA device visible."

# 2) Load pipeline with recommended settings
pipe = DiffusionPipeline.from_pretrained(
	"stabilityai/stable-diffusion-xl-base-1.0",
	torch_dtype=torch.float16,
	use_safetensors=True,
	variant="fp16"
)
pipe.to("cuda")

# 3) Enable memory-efficient attention if needed
# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

prompt = "A basket full of kittens"

# 4) Conservative settings (adjust upward after it works)
# You may want to add height, width, steps, and seed as arguments

image = pipe(prompt=prompt).images[0]

image.save("sdxl.png")
print("Saved sdxl.png on cuda")

# 5) Clean up if running repeatedly in one process
del image, pipe
torch.cuda.empty_cache()

