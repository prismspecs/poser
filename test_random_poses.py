#!/usr/bin/env python3
"""
Random pose testing script.
Randomly selects one image from target_images as target and compares it against comparison_images.
"""

import os
import random
import shutil
from pathlib import Path
import sys

# Add the current directory to Python path so we can import our modules
sys.path.append(str(Path(__file__).parent))

from main import main as run_pose_estimation


def get_target_images():
    """Get all image files from the target_images directory."""
    target_dir = Path("data/target_images")

    if not target_dir.exists():
        print(f"❌ Target images directory not found: {target_dir}")
        return []

    # Get all image files (excluding .avi and other non-image files)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    image_files = []

    for file_path in target_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_files.append(file_path)

    return image_files


def main():
    """Main function to randomly test poses."""
    print("🎲 Random Pose Testing")
    print("=" * 40)

    # Get all target images
    target_images = get_target_images()

    if not target_images:
        print("❌ No target images found!")
        print("💡 Please add some images to data/target_images/")
        return

    print(f"📁 Found {len(target_images)} target images:")
    for i, img in enumerate(target_images, 1):
        print(f"   {i}. {img.name}")

    # Randomly select one as target
    target_image = random.choice(target_images)
    print(f"\n🎯 Randomly selected target: {target_image.name}")

    # Use the comparison_images directory directly
    comparison_dir = Path("data/comparison_images")
    if not comparison_dir.exists():
        print(f"❌ Comparison images directory not found: {comparison_dir}")
        return

    # Count comparison images
    comparison_images = (
        list(comparison_dir.glob("*.jpg"))
        + list(comparison_dir.glob("*.jpeg"))
        + list(comparison_dir.glob("*.png"))
    )

    print(f"📊 Found {len(comparison_images)} comparison images in database")

    # Run pose estimation
    print(f"\n🚀 Running pose estimation...")
    print(f"   Target: {target_image}")
    print(f"   Comparison dir: {comparison_dir}")

    # Set up sys.argv to simulate command line arguments
    sys.argv = [
        "test_random_poses.py",
        "--target",
        str(target_image),
        "--comparison-dir",
        str(comparison_dir),
        "--verbose",
        "--visualize",
    ]

    try:
        run_pose_estimation()
    except Exception as e:
        print(f"❌ Error during pose estimation: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
