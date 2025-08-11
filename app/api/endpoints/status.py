"""
Status endpoint for TinyIMGserver.

This module provides detailed server status information including
version, uptime, and resource utilization.
"""

from fastapi import APIRouter
from app.core.resources import GPUS
from app.core.stats import server_stats
from app.__version__ import __version__, __title__

router = APIRouter()

@router.get("/status")
def get_status():
    """
    Get detailed server status information.
    
    Returns:
        dict: Comprehensive server status including version and resources
    """
    stats = server_stats.get_stats()
    
    return {
        "status": "running",
        "application": __title__,
        "version": __version__,
        "gpus": GPUS,
        "gpu_count": len(GPUS),
        "stats": stats
    }
