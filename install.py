#!/usr/bin/env python3
"""
Installation script for Civitai Randomizer Extension
for Stable Diffusion Forge
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def find_forge_installation():
    """Try to find Stable Diffusion Forge installation directory"""
    common_paths = [
        "stable-diffusion-webui-forge",
        "forge",
        "webui-forge",
        "stable-diffusion-forge",
        "../stable-diffusion-webui-forge",
        "../forge",
        "../../stable-diffusion-webui-forge",
        "../../forge"
    ]
    
    for path in common_paths:
        if os.path.exists(path) and os.path.isdir(path):
            # Check if it looks like a Forge installation
            if (os.path.exists(os.path.join(path, "launch.py")) or 
                os.path.exists(os.path.join(path, "webui.py"))):
                return os.path.abspath(path)
    
    return None

def install_requirements():
    """Install required Python packages"""
    requirements = ["requests>=2.25.1", "typing-extensions>=4.0.0"]
    
    print("Installing required packages...")
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package}")
            return False
    
    return True

def copy_extension_files(forge_path):
    """Copy extension files to Forge extensions directory"""
    extensions_dir = os.path.join(forge_path, "extensions")
    
    # Create extensions directory if it doesn't exist
    os.makedirs(extensions_dir, exist_ok=True)
    
    # Create extension subdirectory
    extension_dir = os.path.join(extensions_dir, "civitai-randomizer")
    os.makedirs(extension_dir, exist_ok=True)
    
    # Files to copy
    files_to_copy = [
        "civitai_randomizer.py",
        "config.json",
        "requirements.txt",
        "README.md"
    ]
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    for file in files_to_copy:
        src = os.path.join(current_dir, file)
        dst = os.path.join(extension_dir, file)
        
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"✓ Copied {file}")
        else:
            print(f"✗ Could not find {file}")
    
    return extension_dir

def main():
    print("Civitai Randomizer Extension Installer")
    print("=====================================")
    
    # Try to find Forge installation
    forge_path = find_forge_installation()
    
    if not forge_path:
        print("\n❌ Could not automatically find Stable Diffusion Forge installation.")
        print("Please enter the path to your Forge installation:")
        forge_path = input("Forge path: ").strip()
        
        if not os.path.exists(forge_path):
            print(f"❌ Path '{forge_path}' does not exist.")
            sys.exit(1)
    
    print(f"✓ Found Forge installation at: {forge_path}")
    
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements. Please install manually:")
        print("pip install requests>=2.25.1 typing-extensions>=4.0.0")
        response = input("Continue anyway? (y/N): ").lower()
        if response != 'y':
            sys.exit(1)
    
    # Copy extension files
    extension_dir = copy_extension_files(forge_path)
    print(f"✓ Extension installed to: {extension_dir}")
    
    print("\n🎉 Installation completed successfully!")
    print("\nNext steps:")
    print("1. Restart Stable Diffusion Forge")
    print("2. Look for 'Civitai Randomizer' section in the interface")
    print("3. (Optional) Get a Civitai API key for better performance")
    print("4. Configure your settings and start generating!")
    
    print("\nFor detailed usage instructions, see README.md")

if __name__ == "__main__":
    main() 