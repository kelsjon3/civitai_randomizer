#!/usr/bin/env python3
"""
Package all Civitai Randomizer files for easy GitHub upload
"""

import os
import shutil
import zipfile
from pathlib import Path

def create_upload_package():
    """Create a directory with all files ready for GitHub upload"""
    
    # Create upload directory
    upload_dir = "civitai_randomizer_upload"
    if os.path.exists(upload_dir):
        shutil.rmtree(upload_dir)
    os.makedirs(upload_dir)
    
    # Files to package
    files_to_copy = [
        "civitai_randomizer.py",
        "config.json", 
        "requirements.txt",
        "README.md",
        "install.py"
    ]
    
    print("📦 Packaging Civitai Randomizer for GitHub upload...")
    
    for file in files_to_copy:
        if os.path.exists(file):
            shutil.copy2(file, os.path.join(upload_dir, file))
            print(f"✓ Added {file}")
        else:
            print(f"✗ Missing {file}")
    
    # Create a zip file too for convenience
    with zipfile.ZipFile("civitai_randomizer_upload.zip", 'w') as zipf:
        for file in files_to_copy:
            if os.path.exists(file):
                zipf.write(file)
    
    print(f"\n🎉 Package created!")
    print(f"📁 Directory: {upload_dir}/")
    print(f"📦 Zip file: civitai_randomizer_upload.zip")
    print(f"\nTo upload to GitHub:")
    print(f"1. Go to: https://github.com/kelsjon3/civitai_randomizer")
    print(f"2. Click 'Add file' → 'Upload files'")
    print(f"3. Drag files from '{upload_dir}' folder")
    print(f"4. Or upload the zip file and extract")

if __name__ == "__main__":
    create_upload_package() 