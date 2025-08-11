#!/usr/bin/env python3
"""
Apple Silicon model download script for TinyIMGserver.

This script downloads FLUX and SDXL models optimized for Apple Silicon.
"""

import sys
import torch
import subprocess

def check_hf_authentication():
    """Check if user has HF_TOKEN environment variable set and valid."""
    import os
    
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        print("❌ HF_TOKEN environment variable not set")
        print("\n📝 Please set up your Hugging Face token:")
        print("1. Visit https://huggingface.co/settings/tokens")
        print("2. Create a new token with 'Read' permissions")
        print("3. Set the environment variable:")
        print("   export HF_TOKEN='your_token_here'")
        print("4. Accept the FLUX license at:")
        print("   https://huggingface.co/black-forest-labs/FLUX.1-schnell")
        return False
    
    try:
        from huggingface_hub import login, whoami
        login(token=hf_token)
        user_info = whoami()
        print(f"✅ Authenticated as: {user_info['name']}")
        
        # Test access to a gated model
        from huggingface_hub import repo_exists
        if repo_exists("black-forest-labs/FLUX.1-schnell", token=hf_token):
            print("✅ Access to FLUX model confirmed")
            return True
        else:
            print("❌ Cannot access FLUX model - make sure you've accepted the license")
            return False
            
    except ImportError:
        print("❌ huggingface_hub not installed. Run: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        print("Please check your HF_TOKEN is valid and you've accepted the FLUX license")
        return False

def check_requirements():
    """Check if required packages are installed."""
    try:
        import torch
        import diffusers
        import transformers
        print(f"✅ PyTorch {torch.__version__}")
        print(f"✅ Diffusers {diffusers.__version__}")
        print(f"✅ Transformers {transformers.__version__}")
        
        if torch.backends.mps.is_available():
            print("✅ MPS acceleration available")
        else:
            print("⚠️  MPS not available - using CPU")
            
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: python3 setup_apple_silicon.py")
        return False

def download_flux():
    """Download FLUX model for Apple Silicon."""
    print("\n🚀 Downloading FLUX.1-schnell model...")
    
    try:
        from diffusers import FluxPipeline
        
        # Determine device
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.float16 if device == "mps" else torch.float32
        
        print(f"Target device: {device}")
        print(f"Data type: {dtype}")
        
        # Download and cache the model
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=dtype
        )
        
        print("✅ FLUX model downloaded successfully!")
        
        # Test generation
        print("🧪 Testing FLUX generation...")
        with torch.no_grad():
            test_image = pipe(
                "a simple test",
                height=256,
                width=256,
                num_inference_steps=1
            ).images[0]
        
        print("✅ FLUX test generation successful!")
        return True
        
    except Exception as e:
        print(f"❌ FLUX download failed: {e}")
        return False

def download_sdxl():
    """Download SDXL model for Apple Silicon."""
    print("\n🚀 Downloading Stable Diffusion XL model...")
    
    try:
        from diffusers import StableDiffusionXLPipeline
        
        # Determine device
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.float16 if device == "mps" else torch.float32
        
        print(f"Target device: {device}")
        print(f"Data type: {dtype}")
        
        # Download and cache the model
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=dtype,
            use_safetensors=True
        )
        
        print("✅ SDXL model downloaded successfully!")
        
        # Test generation
        print("🧪 Testing SDXL generation...")
        with torch.no_grad():
            test_image = pipe(
                "a simple test",
                height=256,
                width=256,
                num_inference_steps=1
            ).images[0]
        
        print("✅ SDXL test generation successful!")
        return True
        
    except Exception as e:
        print(f"❌ SDXL download failed: {e}")
        return False

def main():
    print("🍎 TinyIMGserver Apple Silicon Model Downloader")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check Hugging Face authentication
    if not check_hf_authentication():
        sys.exit(1)
    
    # Show disk space warning
    print("\n💾 Disk Space Requirements:")
    print("- FLUX.1-schnell: ~10GB (requires HF authentication)")
    print("- Stable Diffusion XL: ~7GB")
    print("- Total: ~17GB")
    
    response = input("\nDownload models? (y/N): ").strip().lower()
    if response != 'y':
        print("Download cancelled.")
        return
    
    success_count = 0
    
    # Download FLUX
    if download_flux():
        success_count += 1
    
    # Download SDXL
    if download_sdxl():
        success_count += 1
    
    print(f"\n🎯 Summary: {success_count}/2 models downloaded successfully")
    
    if success_count == 2:
        print("🎉 All models ready! You can now:")
        print("1. Start the server: python3 -m uvicorn app.main:app --reload")
        print("2. Open http://localhost:8000")
        print("3. Generate images with Apple Silicon acceleration!")
    elif success_count == 1:
        print("⚠️  Some models failed to download. Check the errors above.")
    else:
        print("❌ No models downloaded successfully. Check the errors above.")

if __name__ == "__main__":
    main()
