
"""
Main FastAPI application for TinyIMGserver.

This module serves as the entry point for the image generation server,
providing the main FastAPI application instance and core endpoints.
"""

import logging
import os
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Import routers
from app.api.endpoints import generate, models, status, health
from app.core.resources import GPUS
from app.core.stats import server_stats
from app.__version__ import __version__, __title__, __description__, __author__

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log startup information
logger.info("Starting %s v%s", __title__, __version__)

# Determine the correct paths for static files and templates
current_dir = Path(__file__).parent
project_root = current_dir.parent

# Check if running from app directory or project root
if (current_dir / "static").exists():
    static_dir = str(current_dir / "static")
    templates_dir = str(current_dir / "templates")
elif (project_root / "app" / "static").exists():
    static_dir = str(project_root / "app" / "static")
    templates_dir = str(project_root / "app" / "templates")
else:
    # Fallback - create relative paths
    static_dir = "static"
    templates_dir = "templates"

# Create FastAPI application
app = FastAPI(
    title=__title__,
    description=__description__,
    version=__version__
)

# Mount static files (only if directory exists)
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    logger.info("Mounted static files from: %s", static_dir)

# Setup templates (only if directory exists)
if os.path.exists(templates_dir):
    templates = Jinja2Templates(directory=templates_dir)
    logger.info("Using templates from: %s", templates_dir)
else:
    templates = None
    logger.warning("Templates directory not found: %s", templates_dir)

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """
    Serve the main web interface with modern responsive design.
    
    Returns:
        HTMLResponse: Modern web interface for image generation
    """
    if templates is None:
        # Return a simple HTML response when templates are not available
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head>
            <title>TinyIMGserver</title>
        </head>
        <body>
            <h1>TinyIMGserver</h1>
            <p>Server is running! Templates not found - please run from project root.</p>
            <p>API documentation available at <a href="/docs">/docs</a></p>
        </body>
        </html>
        """)
    
    stats = server_stats.get_stats()
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "title": __title__,
        "version": __version__,
        "description": __description__,
        "author": __author__,
        "gpus": GPUS,
        "gpu_count": len(GPUS),
        "stats": stats
    })

app.include_router(generate.router)
app.include_router(models.router)
app.include_router(status.router)
app.include_router(health.router)
