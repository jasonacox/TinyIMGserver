# TinyIMGserver Releases

## v1.0.0 - Initial Release (August 10, 2025)

### Features
- 🚀 **FastAPI Web Server**: Complete REST API with automatic documentation
- 🎨 **Multi-Model Support**: FLUX.1-schnell and Stable Diffusion XL integration
- 🔧 **Intelligent GPU Management**: Thread-safe GPU resource allocation and detection
- 📱 **Cross-Platform Support**: NVIDIA GPUs, Apple Silicon, and CPU fallback
- ✅ **Comprehensive Validation**: Pydantic V2 models with field validators
- 🛡️ **Robust Error Handling**: Graceful error responses and detailed logging
- 🧪 **Complete Test Suite**: 10 test cases with 100% endpoint coverage
- 📚 **Full Documentation**: API docs, code docstrings, and user guides

### API Endpoints
- `POST /generate` - Generate images using AI models
- `GET /health` - Health check endpoint
- `GET /status` - Detailed server status with GPU information
- `GET /models` - List available AI models
- `GET /` - Welcome page with server information
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation (ReDoc)

### Supported Models
- **FLUX.1-schnell**: Fast image generation with excellent quality
- **Stable Diffusion XL**: High-resolution image generation

### Configuration
- YAML-based configuration system
- Configurable GPU timeout settings
- Model-specific configuration options

### Development Features
- Comprehensive test suite with pytest
- Docker containerization support
- Docker Compose for easy deployment
- Development environment setup
- Code quality with linting and formatting

### System Requirements
- Python 3.8+
- PyTorch and Diffusers libraries
- CUDA-compatible GPU (optional)
- 4GB+ RAM (8GB+ recommended)

### Installation & Usage
```bash
# Clone repository
git clone https://github.com/jasonacox/TinyIMGserver.git
cd TinyIMGserver

# Install dependencies
pip install -r requirements.txt

# Start server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Docker Support
```bash
# Build and run with Docker
docker build -t tinyimgserver .
docker run -p 8000:8000 --gpus all tinyimgserver

# Or use Docker Compose
docker-compose up
```

### Testing
- All tests passing: 10/10 ✅
- API endpoint coverage: 100%
- Error handling validation
- Parameter validation tests

### Documentation
- Complete README with setup instructions
- API documentation available at `/docs`
- Comprehensive code docstrings
- Development plan (PLAN.md)

### Known Issues
- None reported in initial release

### Future Enhancements
- Additional model support (planned)
- Performance optimizations
- Enhanced monitoring and metrics
- Batch processing capabilities

---

## Development History

### Pre-Release Development
- Project planning and architecture design
- FastAPI application structure
- GPU resource management implementation
- Model handler development
- API validation and error handling
- Comprehensive testing
- Documentation and containerization

### Contributors
- Jason Cox (@jasonacox) - Project creator and maintainer

### License
MIT License - see LICENSE file for details
