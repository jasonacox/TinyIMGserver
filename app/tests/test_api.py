
"""
Test suite for TinyIMGserver API endpoints.

This module contains comprehensive tests for all API endpoints including:
- Root endpoint (/) - welcome page
- Status endpoint (/status) - server and resource status
- Generate endpoint (/generate) - image generation with validation
- Parameter validation and error handling
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    """Test the root endpoint renders the web interface with version info."""
    response = client.get("/")
    assert response.status_code == 200
    content = response.text
    assert "<title>TinyIMGserver v1.0.0</title>" in content
    assert "Created by Jason Cox" in content

def test_status():
    """Test the status endpoint returns server status and resources."""
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "gpus" in data
    assert "version" in data
    assert "application" in data
    assert "gpu_count" in data
    assert data["application"] == "TinyIMGserver"
    assert data["version"] == "1.0.0"

def _check_generated(response, model):
    # 200 (real model or mock fallback when torch/diffusers are absent) or 503 (no GPU free)
    assert response.status_code in (200, 503)
    if response.status_code == 200:
        data = response.json()
        assert data["image"]
        meta = data["metadata"]
        assert meta["model"] == model
        assert (meta["width"], meta["height"]) == (256, 256)
        assert meta["seed"] == 42

def test_generate_flux_valid():
    """Test valid image generation request for Flux model."""
    payload = {"prompt": "A basket of kittens", "model": "Flux", "width": 256, "height": 256, "steps": 6, "seed": 42}
    _check_generated(client.post("/generate", json=payload), "flux")

def test_generate_sdxl_valid():
    """Test valid image generation request for SDXL model."""
    payload = {"prompt": "A basket of puppies", "model": "SDXL", "width": 256, "height": 256, "steps": 6, "seed": 42}
    _check_generated(client.post("/generate", json=payload), "sdxl")

BASE = {"prompt": "A basket of kittens", "model": "Flux", "width": 256, "height": 256, "steps": 6, "seed": 42}

@pytest.mark.parametrize("override", [
    {"prompt": ""},
    {"model": ""},
    {"model": "UnknownModel"},
    {"width": 32},
    {"height": 4096},
    {"steps": 0},
    {"guidance_scale": 0.5},
    {"seed": -1},
])
def test_generate_invalid(override):
    """Invalid parameters are rejected by request validation (422)."""
    response = client.post("/generate", json={**BASE, **override})
    assert response.status_code == 422
