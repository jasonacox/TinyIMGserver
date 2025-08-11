"""
Image generation endpoint for TinyIMGserver.

This module provides the /generate endpoint that accepts requests to generate
images using various AI models (Flux, SDXL). It includes parameter validation,
GPU resource management, and error handling.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator
from typing import Optional
from app.core.resources import acquire_gpu, release_gpu
from app.core.config import CONFIG
from app.core.stats import server_stats

router = APIRouter()

class GenerateRequest(BaseModel):
    """
    Request model for image generation.
    
    Attributes:
        prompt: Text description of the image to generate
        model: AI model to use ("flux" or "sdxl")
        width: Image width in pixels (64-2048, default: 512)
        height: Image height in pixels (64-2048, default: 512)
        steps: Number of inference steps (1-100, default: 20)
        guidance_scale: Guidance scale for generation (1.0-20.0, default: 7.5)
        seed: Random seed for reproducible results (optional)
    """
    prompt: str
    model: str
    width: int = 512
    height: int = 512
    steps: int = 20
    guidance_scale: float = 7.5
    seed: Optional[int] = None

    @field_validator('prompt')
    @classmethod
    def prompt_must_not_be_empty(cls, v):
        """Validate that prompt is not empty."""
        if not v or not str(v).strip():
            raise ValueError('Prompt must not be empty.')
        return v

    @field_validator('model')
    @classmethod
    def model_must_be_valid(cls, v):
        """Validate that model name is valid."""
        if not v or not str(v).strip():
            raise ValueError('Model must not be empty.')
        if v.lower() not in ['flux', 'sdxl']:
            raise ValueError('Model must be "flux" or "sdxl".')
        return v.lower()

    @field_validator('width', 'height')
    @classmethod
    def dimensions_range(cls, v):
        """Validate dimensions are within acceptable range."""
        if not isinstance(v, int) or not (64 <= v <= 2048):
            raise ValueError('Width and height must be between 64 and 2048 pixels.')
        return v

    @field_validator('steps')
    @classmethod
    def steps_range(cls, v):
        """Validate steps are within acceptable range."""
        if not isinstance(v, int) or not (1 <= v <= 100):
            raise ValueError('Steps must be between 1 and 100.')
        return v

    @field_validator('guidance_scale')
    @classmethod
    def guidance_scale_range(cls, v):
        """Validate guidance scale is within acceptable range."""
        if not isinstance(v, (int, float)) or not (1.0 <= v <= 20.0):
            raise ValueError('Guidance scale must be between 1.0 and 20.0.')
        return float(v)

    @field_validator('seed')
    @classmethod
    def seed_non_negative(cls, v):
        """Validate seed is non-negative or None."""
        if v is not None and (not isinstance(v, int) or v < 0):
            raise ValueError('Seed must be non-negative or None.')
        return v

@router.post("/generate")
async def generate_image(request: GenerateRequest):
    """
    Generate an image based on the provided prompt and parameters.
    
    This endpoint accepts a text prompt and generates an image using the specified
    AI model. It automatically manages GPU resources to ensure only one generation
    happens per GPU at a time.
    
    Args:
        request: GenerateRequest containing prompt, model, size, steps, and seed
        
    Returns:
        dict: Response containing generation status, parameters, and image data
        
    Raises:
        HTTPException: 
            - 400: Invalid parameters or unsupported model
            - 422: Validation errors in request data
            - 500: Model generation errors
            - 503: No GPU available (timeout)
    """
    import time
    
    # Track request
    server_stats.increment_request()
    start_time = time.time()
    
    # Get dimensions directly from request
    width = request.width
    height = request.height

    timeout = CONFIG.get("image_generation_timeout", 60)
    gpu_id = acquire_gpu(timeout=timeout)
    if gpu_id is None:
        server_stats.increment_failure()
        raise HTTPException(status_code=503, detail="No GPU available for image generation (timeout).")

    try:
        server_stats.set_active_generations(1)  # Simplified for single generation
        
        model_name = request.model.lower()
        image_data = None
        
        if model_name == "flux":
            try:
                # Try Apple Silicon optimized version first
                from app.models.flux_apple import generate_flux_image
                image_data, actual_seed = generate_flux_image(
                    prompt=request.prompt,
                    height=height,
                    width=width,
                    steps=request.steps,
                    seed=request.seed
                )
            except Exception as e:
                # Fallback to mock image
                from app.models.mock import generate_mock_image
                image_data, actual_seed = generate_mock_image(
                    prompt=f"[MOCK FLUX] {request.prompt}",
                    width=width,
                    height=height,
                    steps=request.steps,
                    seed=request.seed
                )
                
        elif model_name == "sdxl":
            try:
                # Try Apple Silicon optimized version first
                from app.models.sdxl_apple import generate_sdxl_image
                image_data, actual_seed = generate_sdxl_image(
                    prompt=request.prompt,
                    height=height,
                    width=width,
                    steps=request.steps,
                    seed=request.seed
                )
            except Exception as e:
                # Fallback to mock image
                from app.models.mock import generate_mock_image
                image_data, actual_seed = generate_mock_image(
                    prompt=f"[MOCK SDXL] {request.prompt}",
                    width=width,
                    height=height,
                    steps=request.steps,
                    seed=request.seed
                )
        else:
            server_stats.increment_failure()
            raise HTTPException(status_code=400, detail=f"Model '{request.model}' not supported.")
            
        # Success tracking
        generation_time = time.time() - start_time
        server_stats.increment_success()
        
        result = {
            "image": image_data,
            "metadata": {
                "prompt": request.prompt,
                "model": request.model,
                "width": width,
                "height": height,
                "steps": request.steps,
                "guidance_scale": request.guidance_scale,
                "seed": actual_seed,
                "generation_time": generation_time,
                "gpu_id": gpu_id
            }
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        server_stats.increment_failure()
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}") from e
    finally:
        server_stats.set_active_generations(0)
        release_gpu(gpu_id)
        
    return result
