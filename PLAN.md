# Project Plan: TinyIMGserver

## Overview
TinyIMGserver is a Python FastAPI server for generating images using various models (initially SDXL and Flux). The server exposes APIs for image generation, model listing, status, and health checks. It is designed for easy extensibility, resource management, and containerized deployment on GPU systems.

## Goals
- Provide a RESTful API for image generation (JSON payload: prompt, model, size, etc.)
- Support multiple models, defined in a `config.yaml` for easy extension
- Efficiently manage GPU and CPU resources
- Offer endpoints for listing available models, server status, and health
- Ensure test coverage for all APIs
- Provide clear documentation (README) for setup, configuration, and usage
- Enable containerized deployment via Docker

## Implementation Plan
1. **API Design**
    - `/generate`: Accepts JSON payload to generate images
    - `/models`: Lists available models
    - `/status`: Returns server status and resource usage
    - `/health`: Health check endpoint
2. **Model Integration**
    - Start with SDXL and Flux (see `experiments/` for examples)
    - Use `config.yaml` to define available models and their specifics
    - Design model handler interface for easy extension
3. **Resource Management**
    - Detect and manage available GPUs and CPU memory
    - Assign resources per request, handle contention and fallback to CPU if needed
4. **Configuration**
    - `config.yaml` for models, resources, and server settings
5. **Testing**
    - Implement unit and integration tests for all API endpoints
    - Ensure coverage for model loading, resource management, and error handling
6. **Documentation**
    - Write a comprehensive README with API details, setup instructions, config explanation, and usage examples
7. **Deployment**
    - Create a Dockerfile for containerized deployment on GPU systems
    - Document deployment steps in README

## Open Questions
- What authentication or rate limiting (if any) is required?
- Should generated images be stored or only returned in responses?
- Preferred image formats and maximum sizes?
- Any specific logging/monitoring requirements?
- Target platforms for deployment (cloud, on-prem, etc.)?

## Next Steps
- Finalize requirements and open questions
- Design API schema and config.yaml structure
- Begin implementation of FastAPI server and model handlers
- Set up testing framework
- Draft README and Dockerfile
