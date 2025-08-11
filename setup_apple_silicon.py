#!/usr/bin/env python3
"""
Apple Silicon setup script for TinyIMGserver.

This script sets up the environment for running FLUX and SDXL models
on Apple Silicon Macs with MPS acceleration.
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and show progress."""
    print(f"\n🔧 {description}")
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False

def check_system():
    """Check if we're on Apple Silicon."""
    import platform
    system = platform.system()
    machine = platform.machine()
    
    print(f"System: {system} {machine}")
    
    if system != "Darwin":
        print("❌ This script is designed for macOS only")
        return False
        
    if machine != "arm64":
        print("⚠️  This script is optimized for Apple Silicon (arm64)")
        print("   It may still work on Intel Macs but performance will be different")
    
    return True

def install_dependencies():
    """Install PyTorch and other dependencies optimized for Apple Silicon."""
    
    # Install PyTorch with MPS support
    pytorch_cmd = "pip3 install torch torchvision torchaudio"
    if not run_command(pytorch_cmd, "Installing PyTorch with MPS support"):
        return False
    
    # Install diffusers and transformers
    ml_cmd = "pip3 install diffusers transformers accelerate"
    if not run_command(ml_cmd, "Installing ML libraries"):
        return False
    
    # Install supporting libraries
    support_cmd = "pip3 install huggingface_hub safetensors pillow"
    if not run_command(support_cmd, "Installing supporting libraries"):
        return False
    
    # Install web framework and testing dependencies
    web_cmd = "pip3 install fastapi uvicorn jinja2 httpx pytest"
    if not run_command(web_cmd, "Installing web framework and testing tools"):
        return False
    
    # Install Hugging Face CLI
    hf_cmd = "pip3 install 'huggingface_hub[cli]'"
    if not run_command(hf_cmd, "Installing Hugging Face CLI"):
        return False
    
    return True

def check_hf_auth():
    """Check if HF_TOKEN environment variable is set."""
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        print("❌ HF_TOKEN environment variable not set")
        print("\n📝 To set up Hugging Face authentication:")
        print("1. Visit https://huggingface.co/settings/tokens")
        print("2. Create a new token with 'Read' permissions")
        print("3. Set the environment variable:")
        print("   export HF_TOKEN='your_token_here'")
        print("4. Add to your shell profile to make it permanent:")
        print('   echo \'export HF_TOKEN="your_token_here"\' >> ~/.zshrc')
        print("   source ~/.zshrc")
        return False
    
    try:
        from huggingface_hub import login, whoami
        login(token=hf_token)
        user_info = whoami()
        print(f"✅ Authenticated as: {user_info['name']}")
        return True
    except ImportError:
        print("❌ huggingface_hub not installed")
        return False
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        print("Please check your HF_TOKEN is valid")
        return False

def test_mps():
    """Test if MPS is available."""
    print("\n🧪 Testing MPS availability...")
    
    test_script = '''
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
if torch.backends.mps.is_available():
    print("✅ MPS is available and ready for GPU acceleration!")
    # Test basic operation
    x = torch.randn(3, 3).to("mps")
    y = torch.randn(3, 3).to("mps")
    z = x @ y
    print(f"✅ MPS basic operations working: {z.shape}")
else:
    print("⚠️  MPS not available - will use CPU")
'''
    
    try:
        result = subprocess.run([sys.executable, "-c", test_script], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ MPS test failed: {e}")
        print(f"stderr: {e.stderr}")
        return False

def main():
    print("🍎 TinyIMGserver Apple Silicon Setup")
    print("=" * 40)
    
    # Check system
    if not check_system():
        sys.exit(1)
    
    # Ask for confirmation
    print("\nThis will install/upgrade the following packages:")
    print("- PyTorch (with MPS support)")
    print("- diffusers")
    print("- transformers")
    print("- accelerate")
    print("- huggingface_hub")
    print("- safetensors")
    print("- pillow")
    
    response = input("\nProceed with installation? (y/N): ").strip().lower()
    if response != 'y':
        print("Installation cancelled.")
        return
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Installation failed!")
        sys.exit(1)
    
    # Test MPS
    if not test_mps():
        print("\n⚠️  MPS test failed, but installation completed")
    
    # Check Hugging Face authentication
    hf_authenticated = check_hf_auth()
    
    print("\n🎉 Apple Silicon setup completed!")
    print("\nNext steps:")
    if hf_authenticated:
        print("1. Run the model download script: python3 download_models_apple.py")
        print("2. Start the server: python3 -m uvicorn app.main:app --reload")
        print("3. Open http://localhost:8000 in your browser")
    else:
        print("1. Complete Hugging Face authentication: huggingface-cli login")
        print("2. Run the model download script: python3 download_models_apple.py")
        print("3. Start the server: python3 -m uvicorn app.main:app --reload")
        print("4. Open http://localhost:8000 in your browser")

if __name__ == "__main__":
    main()
