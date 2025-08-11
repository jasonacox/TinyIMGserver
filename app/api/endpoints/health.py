"""
Health check endpoint for TinyIMGserver.

This module provides health monitoring endpoints to check if the server
is running and responsive.
"""

from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
def health_check():
    """
    Health check endpoint to verify server is running.
    
    Returns:
        dict: Health status information
    """
    return {"status": "healthy"}
