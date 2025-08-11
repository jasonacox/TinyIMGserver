"""
Models endpoint for TinyIMGserver.

This module provides the /models endpoint that lists available AI models
for image generation.
"""

from fastapi import APIRouter

router = APIRouter()

@router.get("/models")
def list_models():
    """
    List all available AI models for image generation.
    
    Returns:
        dict: Dictionary containing list of available models
    """
    # TODO: Return list of available models
    return {"models": []}
