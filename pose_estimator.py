"""
YOLOv11 Pose Estimator
Handles model initialization and pose keypoint extraction from images.
"""

import cv2
import numpy as np
import os
from typing import List, Tuple, Optional
from ultralytics import YOLO
from pathlib import Path

from utils.pose_utils import PoseData

try:
    from pose_cache import PoseCache
except ImportError:
    # Fallback if cache module is not available
    PoseCache = None


class PoseEstimator:
    """YOLOv11 pose estimation wrapper."""

    # COCO keypoint names for human pose estimation
    KEYPOINT_NAMES = [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        model_size: str = "n",
        use_cache: bool = True,
    ):
        """
        Initialize the pose estimator.

        Args:
            confidence_threshold: Minimum confidence for pose detection
            model_size: YOLO model size ('n', 's', 'm', 'l', 'x')
            use_cache: Whether to use pose caching for speed
        """
        self.confidence_threshold = confidence_threshold
        self.model_size = model_size
        self.model = None
        self.segmentation_model = None  # For body segmentation
        self.use_cache = use_cache
        self.cache = PoseCache() if use_cache else None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the YOLO pose estimation and segmentation models."""
        print(
            "Initializing YOLOv11 models for all tasks (pose estimation + segmentation)..."
        )

        # Initialize pose estimation model (YOLOv11)
        self.model = self._initialize_pose_model()

        # Initialize segmentation model (YOLOv11)
        self.segmentation_model = self._initialize_segmentation_model()

    def _initialize_pose_model(self) -> YOLO:
        """Initialize YOLOv11 pose estimation model with automatic downloading."""
        # Always prioritize the specifically requested model size
        model_name = f"yolo11{self.model_size}-pose.pt"

        # Check if the requested model exists locally
        if os.path.exists(model_name):
            try:
                model = YOLO(model_name)
                print(f"Loaded existing YOLOv11 pose model: {model_name}")
                return model
            except Exception as e:
                print(f"Failed to load existing {model_name}, will download fresh copy")

        # Download the requested model size
        print(f"Downloading YOLOv11 pose model: {model_name}")
        return self._download_yolo_v11_pose_model()

    def _initialize_segmentation_model(self) -> Optional[YOLO]:
        """Initialize YOLOv11 segmentation model with automatic downloading."""
        # Always prioritize the specifically requested model size
        model_name = f"yolo11{self.model_size}-seg.pt"

        # Check if the requested model exists locally
        if os.path.exists(model_name):
            try:
                model = YOLO(model_name)
                print(f"Loaded existing YOLOv11 segmentation model: {model_name}")
                return model
            except Exception as e:
                print(f"Failed to load existing {model_name}, will download fresh copy")

        # Download the requested model size
        print(f"Downloading YOLOv11 segmentation model: {model_name}")
        return self._download_yolo_v11_segmentation_model()

    def extract_poses(self, image: np.ndarray, image_path: str) -> List[PoseData]:
        """
        Extract poses from an image.

        Args:
            image: Input image as numpy array
            image_path: Path to the image file

        Returns:
            List of detected poses
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")

        # Check cache first if enabled
        if self.use_cache and self.cache:
            cached_poses = self.cache.get_cached_poses(image_path)
            if cached_poses is not None:
                return cached_poses

        # Run inference
        results = self.model(image, verbose=False)

        poses = []
        for result in results:
            if result.keypoints is not None and len(result.keypoints) > 0:
                # Process ALL detected people, not just the first one
                all_keypoints = (
                    result.keypoints.data.cpu().numpy()
                )  # Shape: (N, 17, 3) - N people

                for person_idx, keypoints in enumerate(all_keypoints):
                    # Filter keypoints by confidence
                    valid_keypoints = []
                    for i, kp in enumerate(keypoints):
                        if kp[2] >= self.confidence_threshold:
                            valid_keypoints.append(
                                (float(kp[0]), float(kp[1]), float(kp[2]))
                            )
                        else:
                            # Add None for low-confidence keypoints
                            valid_keypoints.append(None)

                    # Get bounding box for this person
                    if result.boxes is not None and len(result.boxes) > person_idx:
                        box = result.boxes.xyxy[person_idx].cpu().numpy()
                        bbox = (
                            float(box[0]),
                            float(box[1]),
                            float(box[2]),
                            float(box[3]),
                        )
                    else:
                        # Fallback bounding box from keypoints
                        bbox = self._calculate_bbox_from_keypoints(valid_keypoints)

                    # Calculate overall confidence as average of valid keypoints
                    valid_confidences = [
                        kp[2] for kp in valid_keypoints if kp is not None
                    ]
                    overall_confidence = (
                        np.mean(valid_confidences) if valid_confidences else 0.0
                    )

                    # Only add poses with reasonable confidence
                    if overall_confidence >= self.confidence_threshold:
                        # Create pose data
                        pose = PoseData(
                            keypoints=valid_keypoints,
                            bounding_box=bbox,
                            confidence_score=overall_confidence,
                            image_path=image_path,
                            pose_id=f"{Path(image_path).stem}_person_{person_idx}",
                        )

                        poses.append(pose)

        # Cache the results if enabled
        if self.use_cache and self.cache:
            self.cache.cache_poses(image_path, poses)

        return poses

    def _calculate_bbox_from_keypoints(
        self, keypoints: List[Optional[Tuple[float, float, float]]]
    ) -> Tuple[float, float, float, float]:
        """Calculate bounding box from valid keypoints."""
        valid_points = [kp for kp in keypoints if kp is not None]

        if not valid_points:
            return (0.0, 0.0, 0.0, 0.0)

        x_coords = [kp[0] for kp in valid_points]
        y_coords = [kp[1] for kp in valid_points]

        x1, x2 = min(x_coords), max(x_coords)
        y1, y2 = min(y_coords), max(y_coords)

        # Add padding
        padding = 20
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = x2 + padding
        y2 = y2 + padding

        return (float(x1), float(y1), float(x2), float(y2))

    def get_keypoint_names(self) -> List[str]:
        """Get the list of keypoint names."""
        return self.KEYPOINT_NAMES.copy()

    def visualize_pose(
        self,
        image: np.ndarray,
        pose: PoseData,
        draw_keypoints: bool = True,
        draw_bbox: bool = True,
    ) -> np.ndarray:
        """
        Visualize pose on an image.

        Args:
            image: Input image
            pose: Pose data to visualize
            draw_keypoints: Whether to draw keypoints
            draw_bbox: Whether to draw bounding box

        Returns:
            Image with pose visualization
        """
        vis_image = image.copy()

        if draw_bbox:
            x1, y1, x2, y2 = pose.bounding_box
            cv2.rectangle(
                vis_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
            )

        if draw_keypoints:
            for i, keypoint in enumerate(pose.keypoints):
                if keypoint is not None:
                    x, y, conf = keypoint
                    # Draw keypoint
                    cv2.circle(vis_image, (int(x), int(y)), 5, (255, 0, 0), -1)
                    # Draw keypoint name
                    cv2.putText(
                        vis_image,
                        str(i),
                        (int(x) + 10, int(y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )

        return vis_image

    def set_confidence_threshold(self, threshold: float):
        """Update confidence threshold."""
        self.confidence_threshold = max(0.0, min(1.0, threshold))

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        if self.model is None:
            return {"status": "not_initialized"}

        return {
            "status": "loaded",
            "model_size": self.model_size,
            "confidence_threshold": self.confidence_threshold,
            "keypoint_count": len(self.KEYPOINT_NAMES),
            "keypoint_names": self.KEYPOINT_NAMES,
        }

    def create_body_mask(
        self, image: np.ndarray, background_color: Tuple[int, int, int] = (255, 0, 255)
    ) -> np.ndarray:
        """
        Create a body mask from an image, keeping human bodies visible and making background magenta.

        Args:
            image: Input image as numpy array
            background_color: Color to use for background (default: magenta)

        Returns:
            Image with human bodies preserved and background set to specified color
        """
        if self.segmentation_model is None:
            print("Warning: Segmentation model not available, returning original image")
            return image

        try:
            # Run segmentation inference
            results = self.segmentation_model(image, verbose=False)

            # Create mask for all detected objects
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

            for result in results:
                if result.masks is not None:
                    # Get the first mask (assuming single object or taking the largest)
                    if len(result.masks) > 0:
                        # Convert mask to numpy array and scale to image dimensions
                        mask_data = result.masks.data[0].cpu().numpy()
                        mask_data = (mask_data * 255).astype(np.uint8)

                        # Resize mask to match image dimensions if needed
                        if mask_data.shape != image.shape[:2]:
                            mask_data = cv2.resize(
                                mask_data, (image.shape[1], image.shape[0])
                            )

                        # Combine with existing mask
                        mask = cv2.bitwise_or(mask, mask_data)

            # Create output image
            output_image = image.copy()

            # Apply mask: keep original image where mask is white, set background color elsewhere
            mask_bool = mask > 127  # Threshold to create boolean mask

            # Set background color where mask is False (no human detected)
            output_image[~mask_bool] = background_color

            return output_image

        except Exception as e:
            print(f"Warning: Failed to create body mask: {e}")
            return image

    def create_pose_specific_mask(
        self,
        image: np.ndarray,
        pose: PoseData,
        background_color: Tuple[int, int, int] = (255, 0, 255),
    ) -> np.ndarray:
        """
        Create a body mask specifically for a detected pose, keeping only that pose visible.

        Args:
            image: Input image as numpy array
            pose: Specific pose data to mask around
            background_color: Color to use for background (default: magenta)

        Returns:
            Image with only the specific pose preserved and background set to specified color
        """
        if self.segmentation_model is None:
            print("Warning: Segmentation model not available, returning original image")
            return image

        try:
            # Get the bounding box of the specific pose
            x1, y1, x2, y2 = pose.bounding_box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Expand the bounding box slightly to ensure we capture the full body
            padding = 20
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image.shape[1], x2 + padding)
            y2 = min(image.shape[0], y2 + padding)

            # Crop the image to the pose region
            pose_region = image[y1:y2, x1:x2]

            # Run segmentation on the pose region
            results = self.segmentation_model(pose_region, verbose=False)

            # Create output image
            output_image = image.copy()

            if results and len(results) > 0 and results[0].masks is not None:
                # Get the mask for the pose region
                mask_data = results[0].masks.data[0].cpu().numpy()
                mask_data = (mask_data * 255).astype(np.uint8)

                # Resize mask to match the pose region dimensions
                if mask_data.shape != pose_region.shape[:2]:
                    mask_data = cv2.resize(
                        mask_data, (pose_region.shape[1], pose_region.shape[0])
                    )

                # Create a full-size mask
                full_mask = np.zeros(image.shape[:2], dtype=np.uint8)

                # Place the pose mask in the correct location
                full_mask[y1:y2, x1:x2] = mask_data

                # Apply mask: keep original image where mask is white, set background color elsewhere
                mask_bool = full_mask > 127
                output_image[~mask_bool] = background_color

            return output_image

        except Exception as e:
            print(f"Warning: Failed to create pose-specific mask: {e}")
            return image

    def _download_yolo_v11_pose_model(self, model_size: str = None) -> YOLO:
        """
        Download YOLOv11 pose estimation model weights.

        Args:
            model_size: Model size ('n' for nano, 's' for small, 'm' for medium, 'l' for large, 'x' for xlarge)

        Returns:
            YOLO model instance
        """
        import requests
        import os

        # YOLOv11 pose model URLs from Ultralytics assets
        model_urls = {
            "n": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt",
            "s": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-pose.pt",
            "m": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-pose.pt",
            "l": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-pose.pt",
            "x": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-pose.pt",
        }

        if model_size is None:
            model_size = self.model_size
        if model_size not in model_urls:
            print(f"Invalid model size: {model_size}. Using '{self.model_size}'.")
            model_size = self.model_size

        model_name = f"yolo11{model_size}-pose.pt"
        model_url = model_urls[model_size]

        try:
            print(f"Downloading {model_name} from {model_url}...")
            print(
                "This may take a few minutes depending on your internet connection..."
            )

            # Download the model
            response = requests.get(model_url, stream=True)
            response.raise_for_status()

            # Save the model file
            with open(model_name, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"Successfully downloaded {model_name}")

            # Load the downloaded model
            model = YOLO(model_name)
            print(f"Loaded downloaded YOLOv11 pose model: {model_name}")
            return model

        except Exception as e:
            print(f"Failed to download {model_name}: {e}")
            raise RuntimeError(f"Could not download YOLOv11 pose model: {e}")

    def _download_yolo_v11_segmentation_model(
        self, model_size: str = None
    ) -> Optional[YOLO]:
        """
        Download YOLOv11 segmentation model weights.

        Args:
            model_size: Model size ('n' for nano, 's' for small, 'm' for medium, 'l' for large, 'x' for xlarge)

        Returns:
            YOLO model instance if successful, None otherwise
        """
        import requests
        import os

        # YOLOv11 segmentation model URLs from Ultralytics assets
        model_urls = {
            "n": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt",
            "s": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-seg.pt",
            "m": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-seg.pt",
            "l": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-seg.pt",
            "x": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-seg.pt",
        }

        if model_size is None:
            model_size = self.model_size
        if model_size not in model_urls:
            print(f"Invalid model size: {model_size}. Using '{self.model_size}'.")
            model_size = self.model_size

        model_name = f"yolo11{model_size}-seg.pt"
        model_url = model_urls[model_size]

        try:
            print(f"Downloading {model_name} from {model_url}...")
            print(
                "This may take a few minutes depending on your internet connection..."
            )

            # Download the model
            response = requests.get(model_url, stream=True)
            response.raise_for_status()

            # Save the model file
            with open(model_name, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"Successfully downloaded {model_name}")

            # Load the downloaded model
            model = YOLO(model_name)
            print(f"Loaded downloaded YOLOv11 segmentation model: {model_name}")
            return model

        except Exception as e:
            print(f"Failed to download {model_name}: {e}")
            raise RuntimeError(f"Could not download YOLOv11 segmentation model: {e}")
