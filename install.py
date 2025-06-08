import subprocess
import sys
import os

def install_requirements():
    """Install required packages for Civitai Randomizer extension"""
    requirements = [
        "requests>=2.25.1"
    ]
    
    for package in requirements:
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"Installed {package}")
        except subprocess.CalledProcessError:
            print(f"Failed to install {package}")

if __name__ == "__main__":
    install_requirements() 