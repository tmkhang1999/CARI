#!/usr/bin/env python
"""
Setup script for the project.
Checks dependencies and creates necessary directories.
"""

import os
import sys
import subprocess


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("ERROR: Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"Python version: {version.major}.{version.minor}.{version.micro} - OK")
    return True


def check_pip():
    """Check if pip is available."""
    try:
        subprocess.run(["pip", "--version"], capture_output=True, check=True)
        print("pip: OK")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: pip not found")
        return False


def install_dependencies():
    """Install required packages."""
    print("\nInstalling dependencies from requirements.txt...")
    try:
        subprocess.run(
            ["pip", "install", "-r", "requirements.txt"],
            check=True
        )
        print("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to install dependencies: {e}")
        return False


def create_directories():
    """Create necessary directories."""
    directories = [
        "data",
        "checkpoints",
        "logs",
        "outputs",
        "outputs/eval",
        "outputs/train"
    ]

    print("\nCreating directories...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  {directory}: OK")

    return True


def verify_imports():
    """Verify that key packages can be imported."""
    print("\nVerifying package imports...")
    packages = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("numpy", "NumPy"),
        ("yaml", "PyYAML"),
        ("PIL", "Pillow"),
        ("tqdm", "tqdm"),
        ("timm", "timm")
    ]

    all_ok = True
    for module_name, package_name in packages:
        try:
            __import__(module_name)
            print(f"  {package_name}: OK")
        except ImportError:
            print(f"  {package_name}: FAILED")
            all_ok = False

    return all_ok


def check_gpu():
    """Check if CUDA is available."""
    print("\nChecking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            print(f"  CUDA available: YES")
            print(f"  GPU count: {device_count}")
            print(f"  GPU 0: {device_name}")
            return True
        else:
            print("  CUDA available: NO (will use CPU)")
            return False
    except ImportError:
        print("  Cannot check GPU (PyTorch not installed)")
        return False


def main():
    """Main setup function."""
    print("="*60)
    print("Multi-Stage Intrinsic Decomposition - Setup")
    print("="*60)

    # Check Python version
    if not check_python_version():
        return False

    # Check pip
    if not check_pip():
        return False

    # Install dependencies
    if not install_dependencies():
        print("\nWARNING: Some dependencies may not have installed correctly")
        print("You may need to install them manually")

    # Create directories
    create_directories()

    # Verify imports
    if not verify_imports():
        print("\nWARNING: Some packages failed to import")
        print("Please check error messages above")
        return False

    # Check GPU
    check_gpu()

    print("\n" + "="*60)
    print("Setup completed successfully!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Run tests: python tests/test_models.py")
    print("  2. Prepare your dataset in datasets/ directory")
    print("  3. Update src/configs/base.yaml with your settings")
    print("  4. Start training: python src/train_stage1.py --config src/configs/base.yaml")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

