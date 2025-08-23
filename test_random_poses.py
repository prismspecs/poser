#!/usr/bin/env python3
"""
Random pose testing script.
Randomly selects one image from test_images as target and compares it against the rest.
"""

import os
import random
import shutil
from pathlib import Path
import sys

# Add the current directory to Python path so we can import our modules
sys.path.append(str(Path(__file__).parent))

from main import main as run_pose_estimation


def get_test_images():
    """Get all image files from the test_images directory."""
    test_dir = Path("data/test_images")

    if not test_dir.exists():
        print(f"❌ Test images directory not found: {test_dir}")
        return []

    # Get all image files (excluding .avi and other non-image files)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    image_files = []

    for file_path in test_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_files.append(file_path)

    return image_files


def main():
    """Main function to randomly test poses."""
    print("🎲 Random Pose Testing")
    print("=" * 40)

    # Get all test images
    test_images = get_test_images()

    if not test_images:
        print("❌ No test images found!")
        print("💡 Please add some images to data/test_images/")
        return

    print(f"📁 Found {len(test_images)} test images:")
    for i, img in enumerate(test_images, 1):
        print(f"   {i}. {img.name}")

    # Randomly select one as target
    target_image = random.choice(test_images)
    print(f"\n🎯 Randomly selected target: {target_image.name}")

    # Create comparison directory with remaining images
    comparison_dir = Path("data/comparison_temp")
    if comparison_dir.exists():
        shutil.rmtree(comparison_dir)
    comparison_dir.mkdir(exist_ok=True)

    # Copy remaining images to comparison directory
    comparison_images = []
    for img in test_images:
        if img != target_image:
            comparison_path = comparison_dir / img.name
            shutil.copy2(img, comparison_path)
            comparison_images.append(comparison_path)

    print(f"📊 Comparing against {len(comparison_images)} images:")
    for img in comparison_images:
        print(f"   - {img.name}")

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
    finally:
        # Clean up temporary comparison directory
        if comparison_dir.exists():
            shutil.rmtree(comparison_dir)
            print(f"\n🧹 Cleaned up temporary files")


if __name__ == "__main__":
    main()
