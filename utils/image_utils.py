"""
Image utility functions for pose estimation.
Handles image loading, validation, and preprocessing.
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supported image formats
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    Load an image from file path.

    Args:
        image_path: Path to the image file

    Returns:
        Image as numpy array in BGR format (OpenCV format)

    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image format is not supported
        RuntimeError: If image cannot be loaded
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    if image_path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported image format: {image_path.suffix}. Supported: {SUPPORTED_FORMATS}"
        )

    try:
        # Try OpenCV first (faster for most formats)
        image = cv2.imread(str(image_path))
        if image is not None:
            return image

        # Fallback to PIL if OpenCV fails
        logger.warning(f"OpenCV failed to load {image_path}, trying PIL...")
        pil_image = Image.open(image_path)

        # Convert PIL image to numpy array
        if pil_image.mode in ("RGBA", "LA", "P"):
            pil_image = pil_image.convert("RGB")

        # Convert to BGR (OpenCV format)
        image_array = np.array(pil_image)
        if len(image_array.shape) == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        return image_array

    except Exception as e:
        raise RuntimeError(f"Failed to load image {image_path}: {e}")


def validate_image_path(image_path: Union[str, Path]) -> bool:
    """
    Validate if an image path is valid and accessible.

    Args:
        image_path: Path to the image file

    Returns:
        True if valid, False otherwise
    """
    try:
        path = Path(image_path)

        # Check if file exists
        if not path.exists():
            return False

        # Check if it's a file
        if not path.is_file():
            return False

        # Check file extension
        if path.suffix.lower() not in SUPPORTED_FORMATS:
            return False

        # Check if file is readable
        if not path.is_file() or not path.stat().st_size > 0:
            return False

        return True

    except Exception:
        return False


def preprocess_image(
    image: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None,
    normalize: bool = True,
) -> np.ndarray:
    """
    Preprocess image for pose estimation.

    Args:
        image: Input image as numpy array
        target_size: Target size (width, height) for resizing
        normalize: Whether to normalize pixel values to [0, 1]

    Returns:
        Preprocessed image
    """
    processed_image = image.copy()

    # Resize if target size specified
    if target_size is not None:
        width, height = target_size
        processed_image = cv2.resize(
            processed_image, (width, height), interpolation=cv2.INTER_AREA
        )

    # Normalize pixel values
    if normalize:
        if processed_image.dtype == np.uint8:
            processed_image = processed_image.astype(np.float32) / 255.0
        elif processed_image.dtype == np.float32:
            # Already normalized
            pass
        else:
            processed_image = processed_image.astype(np.float32)

    return processed_image


def save_image(
    image: np.ndarray, output_path: Union[str, Path], quality: int = 95
) -> bool:
    """
    Save image to file.

    Args:
        image: Image to save as numpy array
        output_path: Output file path
        quality: JPEG quality (1-100) for JPEG files

    Returns:
        True if successful, False otherwise
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure image is in correct format
        if image.dtype == np.float32 and image.max() <= 1.0:
            # Convert from [0,1] to [0,255]
            image = (image * 255).astype(np.uint8)

        # Save using OpenCV
        success = cv2.imwrite(str(output_path), image)

        if not success:
            # Fallback to PIL
            if len(image.shape) == 3:
                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image

            pil_image = Image.fromarray(image_rgb)

            if output_path.suffix.lower() in [".jpg", ".jpeg"]:
                pil_image.save(output_path, "JPEG", quality=quality)
            else:
                pil_image.save(output_path)

        return True

    except Exception as e:
        logger.error(f"Failed to save image to {output_path}: {e}")
        return False


def get_image_info(image_path: Union[str, Path]) -> dict:
    """
    Get basic information about an image.

    Args:
        image_path: Path to the image file

    Returns:
        Dictionary with image information
    """
    try:
        image = load_image(image_path)
        path = Path(image_path)

        info = {
            "path": str(path),
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "dimensions": (image.shape[1], image.shape[0]),  # width, height
            "channels": image.shape[2] if len(image.shape) == 3 else 1,
            "dtype": str(image.dtype),
            "format": path.suffix.lower(),
        }

        return info

    except Exception as e:
        return {"path": str(image_path), "error": str(e)}


def batch_load_images(
    image_paths: list, max_images: Optional[int] = None, show_progress: bool = True
) -> list:
    """
    Load multiple images in batch.

    Args:
        image_paths: List of image paths
        max_images: Maximum number of images to load
        show_progress: Whether to show loading progress

    Returns:
        List of loaded images
    """
    if max_images:
        image_paths = image_paths[:max_images]

    loaded_images = []
    failed_paths = []

    for i, path in enumerate(image_paths):
        if show_progress:
            print(f"Loading image {i+1}/{len(image_paths)}: {Path(path).name}")

        try:
            image = load_image(path)
            loaded_images.append(image)
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
            failed_paths.append(path)

    if failed_paths:
        logger.warning(f"Failed to load {len(failed_paths)} images")

    return loaded_images


def create_image_grid(
    images: list, grid_size: Optional[Tuple[int, int]] = None, max_images: int = 16
) -> np.ndarray:
    """
    Create a grid of images for visualization.

    Args:
        images: List of images
        grid_size: Grid dimensions (rows, cols)
        max_images: Maximum number of images to include

    Returns:
        Grid image as numpy array
    """
    if not images:
        return np.array([])

    # Limit number of images
    images = images[:max_images]

    # Determine grid size if not specified
    if grid_size is None:
        n_images = len(images)
        cols = int(np.ceil(np.sqrt(n_images)))
        rows = int(np.ceil(n_images / cols))
        grid_size = (rows, cols)

    rows, cols = grid_size

    # Get dimensions from first image
    if len(images[0].shape) == 3:
        height, width, channels = images[0].shape
    else:
        height, width = images[0].shape
        channels = 1

    # Create grid
    grid_height = rows * height
    grid_width = cols * width

    if channels == 1:
        grid = np.zeros((grid_height, grid_width), dtype=np.uint8)
    else:
        grid = np.zeros((grid_height, grid_width, channels), dtype=np.uint8)

    # Fill grid
    for i, image in enumerate(images):
        row = i // cols
        col = i % cols

        y_start = row * height
        y_end = y_start + height
        x_start = col * width
        x_end = x_start + width

        # Ensure image fits in grid
        if y_end <= grid_height and x_end <= grid_width:
            grid[y_start:y_end, x_start:x_end] = image

    return grid
