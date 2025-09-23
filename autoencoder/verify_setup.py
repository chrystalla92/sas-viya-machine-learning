#!/usr/bin/env python3
"""
Verification script to test the autoencoder package setup.

This script verifies that:
1. All dependencies can be imported
2. PyTorch can detect available hardware
3. Package structure allows clean imports
4. Basic functionality works as expected
"""

import sys
import subprocess
from pathlib import Path


def check_dependencies():
    """Check that all required dependencies can be imported."""
    print("Checking dependencies...")
    
    required_packages = [
        'torch',
        'torchvision', 
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'tqdm',
        'sklearn',
        'PIL',
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\nFailed to import: {failed_imports}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("All dependencies imported successfully!\n")
    return True


def check_torch_functionality():
    """Test basic PyTorch functionality and hardware detection."""
    print("Checking PyTorch functionality...")
    
    try:
        import torch
        
        # Basic tensor operations
        x = torch.randn(3, 4)
        y = torch.randn(4, 5)
        z = torch.mm(x, y)
        print(f"✓ Basic tensor operations work")
        
        # Device detection
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            
            # Test CUDA operations
            x_gpu = x.cuda()
            y_gpu = y.cuda()
            z_gpu = torch.mm(x_gpu, y_gpu)
            print(f"✓ CUDA operations work")
        else:
            print("ℹ CUDA not available, using CPU")
        
        print("PyTorch functionality verified!\n")
        return True
        
    except Exception as e:
        print(f"✗ PyTorch functionality test failed: {e}")
        return False


def check_package_structure():
    """Verify the package structure allows clean imports."""
    print("Checking package structure...")
    
    try:
        # Add src to Python path for testing
        src_path = Path(__file__).parent / "src"
        sys.path.insert(0, str(src_path))
        
        # Test imports
        import src
        print(f"✓ Main package import works")
        
        # Test submodule imports 
        from src import data, models, utils
        print(f"✓ Submodule imports work")
        
        # Test utility imports
        from src.utils.device_utils import get_device, setup_cuda
        device = get_device()
        info = setup_cuda()
        print(f"✓ Utility function imports work")
        print(f"✓ Device detected: {device}")
        
        print("Package structure verified!\n")
        return True
        
    except Exception as e:
        print(f"✗ Package structure test failed: {e}")
        return False


def check_project_structure():
    """Verify all required directories and files exist."""
    print("Checking project structure...")
    
    base_path = Path(__file__).parent
    
    required_files = [
        "requirements.txt",
        "setup.py",
        ".gitignore",
        "src/__init__.py",
        "src/data/__init__.py",
        "src/models/__init__.py",
        "src/utils/__init__.py",
        "src/scripts/__init__.py",
        "tests/__init__.py",
        "tests/conftest.py",
        "data/README.md",
        "models/README.md",
        "scripts/README.md",
    ]
    
    missing_files = []
    
    for file_path in required_files:
        full_path = base_path / file_path
        if full_path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\nMissing files: {missing_files}")
        return False
    
    print("Project structure verified!\n")
    return True


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("SAS VIYA AUTOENCODER PACKAGE VERIFICATION")
    print("=" * 60)
    
    checks = [
        ("Project Structure", check_project_structure),
        ("Dependencies", check_dependencies),
        ("PyTorch Functionality", check_torch_functionality),
        ("Package Imports", check_package_structure),
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        print(f"\n{'=' * 30}")
        print(f"RUNNING: {check_name}")
        print(f"{'=' * 30}")
        
        results[check_name] = check_func()
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for check_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {check_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL CHECKS PASSED! Package setup is complete.")
        print("\nNext steps:")
        print("1. Install in development mode: pip install -e .")
        print("2. Run tests: pytest tests/")
        print("3. Begin MNIST data processing implementation")
    else:
        print("❌ SOME CHECKS FAILED! Please address the issues above.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
