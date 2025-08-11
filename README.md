# TinyIMGserver

A lightweight FastAPI server for AI image generation with support for multiple models including FLUX.1-schnell and Stable Diffusion XL.

## Features

- 🚀 **Fast API**: Built with FastAPI for high performance and automatic API documentation
- 🎨 **Multiple Models**: Support for FLUX.1-schnell and Stable Diffusion XL
- 🍎 **Apple Silicon Optimized**: Native MPS acceleration for M1-M4 Macs
- 🌐 **Modern Web Interface**: Responsive web UI with real-time generation
- 🔧 **GPU Management**: Intelligent GPU resource management with thread-safe access
- 📱 **Cross-Platform**: Support for NVIDIA GPUs, Apple Silicon, and CPU fallback
- ✅ **Validation**: Comprehensive request validation with Pydantic V2
- 🛡️ **Error Handling**: Robust error handling and logging
- 📊 **Real-time Stats**: Live server statistics and generation monitoring
- 🧪 **Tested**: Comprehensive test suite with 100% endpoint coverage
- 📚 **Documented**: Complete API documentation and code docstrings

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, CPU fallback available)
- PyTorch and Diffusers libraries
- **Hugging Face Account** (required for FLUX model access)

### Hugging Face Authentication

Before installation, set up Hugging Face authentication:

1. Create account at [huggingface.co](https://huggingface.co)
2. Accept FLUX license at [black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell)
3. Get access token from [Settings > Access Tokens](https://huggingface.co/settings/tokens)
4. Login: `pip install huggingface_hub[cli] && huggingface-cli login`

### Installation

1. Clone the repository:
```bash
git clone https://github.com/jasonacox/TinyIMGserver.git
cd TinyIMGserver
```

2. Install dependencies:

**For Apple Silicon Macs (Recommended):**
```bash
python3 setup_apple_silicon.py
```

**For NVIDIA GPUs or General Installation:**
```bash
pip install -r requirements.txt
```

**For CPU-only installation:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install diffusers transformers accelerate huggingface_hub safetensors pillow fastapi uvicorn jinja2
```

3. Configure the server (optional):
```bash
cp config.yaml.example config.yaml
# Edit config.yaml with your preferences
```

4. Start the server:
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The server will be available at `http://localhost:8000`

## Apple Silicon (macOS) Setup

TinyIMGserver has optimized support for Apple Silicon Macs with Metal Performance Shaders (MPS) acceleration. Follow these steps for the best performance:

### Prerequisites

- macOS with Apple Silicon (M1-M4 or newer)
- Python 3.8+ (recommend using miniforge/conda for Apple Silicon)
- Xcode Command Line Tools
- **Hugging Face Account** (required for FLUX model access)

### Hugging Face Setup (Required)

Before downloading models, you need to set up Hugging Face authentication:

1. **Create a Hugging Face account** at [huggingface.co](https://huggingface.co)

2. **Accept the FLUX license** by visiting [black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) and clicking "Accept license"

3. **Get your access token**:
   - Go to [Settings > Access Tokens](https://huggingface.co/settings/tokens)
   - Create a new token with "Read" permissions
   - Copy the token

4. **Set the environment variable**:
```bash
export HF_TOKEN="your_token_here"
```

To make this permanent, add it to your shell profile:
```bash
echo 'export HF_TOKEN="your_token_here"' >> ~/.zshrc
source ~/.zshrc
```

Alternatively, you can copy `.env.example` to `.env` and edit it:
```bash
cp .env.example .env
# Edit .env file with your token
```

### Quick Setup for Apple Silicon

1. Clone the repository:
```bash
git clone https://github.com/jasonacox/TinyIMGserver.git
cd TinyIMGserver
```

2. Run the automated Apple Silicon setup:
```bash
python3 setup_apple_silicon.py
```

This script will:
- Install PyTorch with MPS (Metal Performance Shaders) support
- Install compatible versions of diffusers, transformers, and accelerate
- Test MPS availability and basic GPU operations

3. Download AI models optimized for Apple Silicon:
```bash
python3 download_models_apple.py
```

This will download:
- FLUX.1-schnell (~10GB) - Fast image generation model
- Stable Diffusion XL (~7GB) - High-quality image generation

**Note:** Model downloads require ~17GB of disk space and may take 10-20 minutes depending on your internet connection.

4. Start the server with MPS acceleration:
```bash
python3 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

5. Open the modern web interface:
```
http://localhost:8000
```

### Apple Silicon Performance Tips

- **Memory Management**: Apple Silicon models use unified memory. Close other memory-intensive applications for best performance.
- **Model Selection**: 
  - FLUX: Optimized for speed (1-6 steps), good for rapid prototyping
  - SDXL: Higher quality output (20-50 steps), better for final images
- **Image Sizes**: 
  - FLUX: Works well with 768x768 or 1024x1024
  - SDXL: Optimized for 1024x1024 and larger
- **Batch Processing**: Generate images one at a time for optimal memory usage

### Troubleshooting Apple Silicon

If you encounter issues:

1. **MPS Not Available**: 
```bash
python3 -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

2. **Hugging Face Authentication Issues**:
```bash
# Check if logged in
huggingface-cli whoami

# Login if needed
huggingface-cli login

# Verify FLUX access
huggingface-cli repo info black-forest-labs/FLUX.1-schnell
```

3. **Memory Errors**: Reduce image size or use fewer inference steps

4. **Import Errors**: Reinstall with compatible versions:
```bash
pip install accelerate==0.20.3 transformers==4.44.0
```

5. **Model Download Issues**: 
   - Check internet connection and available disk space (need ~20GB free)
   - Ensure Hugging Face authentication is complete
   - Verify you've accepted the FLUX license agreement

### Apple Silicon Web Interface

The server includes a modern, responsive web interface optimized for all devices:

- **Real-time Generation**: Watch images generate in real-time
- **Model Switching**: Easy switching between FLUX and SDXL
- **Parameter Control**: Intuitive sliders for steps, guidance scale, dimensions
- **Download Support**: One-click download of generated images
- **Mobile Friendly**: Responsive design works on iPhone/iPad
- **Statistics Dashboard**: Monitor server performance and generation history

## API Endpoints

### Web Interface
`GET /`

Access the modern, responsive web interface for interactive image generation. The web interface provides:
- Real-time image generation with live preview
- Intuitive controls for all generation parameters
- Model switching between FLUX and SDXL
- Download generated images
- Server statistics and monitoring
- Mobile-responsive design

Simply navigate to `http://localhost:8000` in your browser after starting the server.

### Generate Image
`POST /generate`

Generate an image using AI models.

**Request Body:**
```json
{
  "prompt": "A beautiful sunset over mountains",
  "model": "flux",
  "width": 512,
  "height": 512,
  "steps": 20,
  "guidance_scale": 7.5,
  "seed": 42
}
```

**Parameters:**
- `prompt` (required): Text description of the image to generate
- `model` (required): Model to use ("flux" or "sdxl")
- `width` (optional): Image width in pixels (64-2048, default: 512)
- `height` (optional): Image height in pixels (64-2048, default: 512)
- `steps` (optional): Number of inference steps (1-100, default: 20)
- `guidance_scale` (optional): Guidance scale (1.0-20.0, default: 7.5)
- `seed` (optional): Random seed for reproducible results

**Response:**
```json
{
  "image": "base64_encoded_image_data",
  "metadata": {
    "model": "flux",
    "prompt": "A beautiful sunset over mountains",
    "width": 512,
    "height": 512,
    "steps": 20,
    "guidance_scale": 7.5,
    "seed": 42,
    "generation_time": 2.34
  }
}
```

### Health Check
`GET /health`

Check if the server is healthy and responsive.

**Response:**
```json
{
  "status": "healthy"
}
```

### Server Status
`GET /status`

Get detailed server status including GPU information.

**Response:**
```json
{
  "status": "running",
  "application": "TinyIMGserver",
  "version": "1.0.0",
  "gpus": [
    {
      "id": 0,
      "name": "NVIDIA GeForce RTX 4090",
      "memory": "24GB",
      "type": "NVIDIA"
    }
  ],
  "gpu_count": 1
}
```

### List Models
`GET /models`

List all available AI models.

**Response:**
```json
{
  "models": ["flux", "sdxl"]
}
```

## Configuration

The server can be configured using `config.yaml`:

```yaml
# GPU timeout for model loading (seconds)
gpu_timeout: 60

# Model-specific settings
models:
  flux:
    enabled: true
    model_id: "black-forest-labs/FLUX.1-schnell"
  
  sdxl:
    enabled: true
    model_id: "stabilityai/stable-diffusion-xl-base-1.0"
```

## Development

### Running Tests

```bash
python -m pytest app/tests/ -v
```

### Project Structure

```
TinyIMGserver/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry point
│   ├── api/
│   │   ├── __init__.py
│   │   └── endpoints/
│   │       ├── __init__.py
│   │       ├── generate.py     # Image generation endpoint
│   │       ├── health.py       # Health check endpoint
│   │       ├── models.py       # Models listing endpoint
│   │       └── status.py       # Status endpoint
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py           # Configuration management
│   │   ├── resources.py        # GPU resource management
│   │   └── stats.py            # Server statistics tracking
│   ├── models/
│   │   ├── __init__.py
│   │   ├── flux.py             # FLUX model handler
│   │   ├── flux_apple.py       # FLUX optimized for Apple Silicon
│   │   ├── sdxl.py             # SDXL model handler
│   │   ├── sdxl_apple.py       # SDXL optimized for Apple Silicon
│   │   └── mock.py             # Mock generation for testing
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css       # Modern web interface styling
│   │   └── js/
│   │       └── app.js          # Interactive web interface
│   ├── templates/
│   │   └── index.html          # Responsive web interface
│   └── tests/
│       ├── __init__.py
│       └── test_api.py         # Comprehensive test suite
├── setup_apple_silicon.py      # Apple Silicon setup script
├── download_models_apple.py    # Apple Silicon model downloader
├── download_models.py          # General model downloader
├── test_flux.py                # Simple FLUX test script
├── requirements-apple-silicon.txt  # Apple Silicon dependencies
└── requirements-models.txt     # Model-specific dependencies
├── experiments/                # Original model experiments
│   ├── flux.py
│   ├── qwen.py
│   └── sdxl.py
├── config.yaml                 # Server configuration
├── requirements.txt            # Python dependencies
├── PLAN.md                     # Development plan
└── README.md                   # This file
```

### Adding New Models

1. Create a new model handler in `app/models/`:
```python
def generate_mymodel_image(prompt: str, **kwargs):
    """Generate image using MyModel."""
    # Implementation here
    pass
```

2. Add the model to `app/api/endpoints/generate.py`:
```python
elif model == "mymodel":
    from app.models.mymodel import generate_mymodel_image
    image_data = generate_mymodel_image(prompt, **params)
```

3. Update validation and tests as needed.

## Docker Support

Build and run with Docker:

```bash
docker build -t tinyimgserver .
docker run -p 8000:8000 --gpus all tinyimgserver
```

## API Documentation

When the server is running, visit:
- Interactive API docs: `http://localhost:8000/docs`
- ReDoc documentation: `http://localhost:8000/redoc`

## Performance

- **GPU Acceleration**: Automatic detection and use of available GPUs
- **Resource Management**: Thread-safe GPU allocation prevents conflicts
- **Memory Efficient**: Models are loaded on-demand and cached
- **Async Processing**: Non-blocking request handling

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce image dimensions or batch size
2. **Model Download Fails**: Check internet connection and Hugging Face access
3. **Import Errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`

### Logging

The server provides detailed logging for debugging:
```bash
python -m uvicorn app.main:app --log-level debug
```

## Running the Server

Once setup is complete, start the server:

```bash
# Make sure HF_TOKEN is set
export HF_TOKEN="your_token_here"

# Start the server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Or run from the app directory:
```bash
cd app
export HF_TOKEN="your_token_here"
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The web interface will be available at http://localhost:8000

For containerized deployments, set the `HF_TOKEN` environment variable:

```bash
# Docker run example
docker run -e HF_TOKEN="your_token_here" -p 8000:8000 tinyimgserver

# Docker Compose example  
services:
  tinyimgserver:
    image: tinyimgserver
    environment:
      - HF_TOKEN=${HF_TOKEN}
    ports:
      - "8000:8000"
```

Make sure to set `HF_TOKEN` in your environment before running docker-compose:
```bash
export HF_TOKEN="your_token_here"
docker-compose up
```

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) for the excellent web framework
- [Hugging Face](https://huggingface.co/) for model hosting and Diffusers library
- [Stability AI](https://stability.ai/) for Stable Diffusion models
- [Black Forest Labs](https://blackforestlabs.ai/) for FLUX models
