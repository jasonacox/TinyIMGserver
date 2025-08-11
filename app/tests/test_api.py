
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
    """Test the root endpoint returns welcome page with version info."""
    response = client.get("/")
    assert response.status_code == 200
    content = response.text
    assert "Welcome to TinyIMGserver" in content
    assert "Version:</strong> 1.0.0" in content
    assert "Author:</strong> Jason Cox" in content

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

def test_generate_flux_valid():
    """Test valid image generation request for Flux model."""
    payload = {
        "prompt": "A basket of kittens",
        "model": "Flux",
        "size": "768x768",
        "steps": 6,
        "seed": 42
    }
    response = client.post("/generate", json=payload)
    # Accept 200 or 500 (if model not installed)
    assert response.status_code in (200, 500, 503)
    if response.status_code == 200:
        data = response.json()
        assert data["model"] == "Flux"
        assert "image_path" in data

def test_generate_sdxl_valid():
    """Test valid image generation request for SDXL model."""
    payload = {
        "prompt": "A basket of puppies",
        "model": "SDXL",
        "size": "768x768",
        "steps": 6,
        "seed": 42
    }
    response = client.post("/generate", json=payload)
    assert response.status_code in (200, 500, 503)
    if response.status_code == 200:
        data = response.json()
        assert data["model"] == "SDXL"
        assert "image_path" in data

@pytest.mark.parametrize("payload,expected_status", [
    ({"prompt": "", "model": "Flux", "size": "768x768", "steps": 6, "seed": 42}, 422),
    ({"prompt": "A basket of kittens", "model": "", "size": "768x768", "steps": 6, "seed": 42}, 422),
    ({"prompt": "A basket of kittens", "model": "Flux", "size": "badsize", "steps": 6, "seed": 42}, 422),
    ({"prompt": "A basket of kittens", "model": "Flux", "size": "768x768", "steps": 0, "seed": 42}, 422),
    ({"prompt": "A basket of kittens", "model": "Flux", "size": "768x768", "steps": 6, "seed": -1}, 422),
    ({"prompt": "A basket of kittens", "model": "UnknownModel", "size": "768x768", "steps": 6, "seed": 42}, 400),
])
def test_generate_invalid(payload, expected_status):
    """Test invalid requests return appropriate error codes."""
    response = client.post("/generate", json=payload)
    assert response.status_code == expected_status