# Pose Estimation and Matching System - Technical Reference

## Project Structure
```
poser/
├── main.py                  # CLI application entry point
├── pose_estimator.py        # YOLO pose estimation wrapper
├── pose_matcher.py          # Pose similarity calculation
├── pose_visualizer.py       # Diagnostic visualization
├── pose_cache.py            # JSON-based pose caching
├── utils/                   # Utility modules
│   ├── image_utils.py       # Image loading and processing
│   └── pose_utils.py        # Pose data structures
├── data/                    # Data directories
└── results/                 # Output results
```

## Core Components

### Pose Estimator (`pose_estimator.py`)
- YOLO v11 model initialization and management
- Pose keypoint extraction from images
- Multi-person detection and handling
- Body segmentation for masking
- Confidence scoring and filtering

### Pose Matcher (`pose_matcher.py`)
- Advanced pose similarity algorithms
- Position, scale, and orientation invariant comparison
- Mean Squared Error (MSE) based similarity scoring
- Best match selection from multiple poses

### Pose Visualizer (`pose_visualizer.py`)
- Diagnostic visualization generation
- Pose skeleton drawing and keypoint display
- Pose alignment and overlay functionality
- Grid-based comparison visualization
- HD resolution output (1920x1080)

### Pose Cache (`pose_cache.py`)
- JSON-based pose data caching system
- Image content hashing for unique identification
- Performance improvement for subsequent runs

## Data Structures

### PoseData
```python
class PoseData:
    keypoints: List[Optional[Tuple[float, float, float]]]  # x, y, confidence
    bounding_box: Tuple[float, float, float, float]        # x1, y1, x2, y2
    confidence_score: float
    image_path: str
    pose_id: str
```

### SimilarityResult
```python
class SimilarityResult:
    target_image: str
    comparison_image: str
    similarity_score: float
    target_pose_id: str
    comparison_pose_id: str
    rank: int
```

## Technical Specifications

### YOLO v11 Keypoints (COCO format)
17 keypoints: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles

### Pose Quality Filtering
- Major body region completeness filtering
- Requires keypoints from all major regions (head, torso, arms, legs)
- Relative visibility threshold (default: 0.65)
- Critical keypoint penalties for incomplete poses
- Increased confidence thresholds for more selective human detection:
  - Main pose detection: 0.7 (was 0.5)
  - Keypoint visibility: 0.5 (was 0.3)

### Visualization Controls
- Configurable skeleton drawing on comparison images
- Configurable body masking on comparison images
- Target overlay always shows skeleton
- HD resolution with consistent sizing

## CLI Interface

### Core Arguments
- `--target`: Target image path or directory
- `--comparison-dir`: Directory containing comparison images
- `--threshold`: Confidence threshold for pose detection
- `--visualize`: Generate diagnostic visualizations

### Visualization Controls
- `--no-mask`: Disable body segmentation masks on comparison images (masks are on by default)
- `--no-skeleton`: Disable skeleton drawing (lines + keypoints) on comparison images

## Dependencies
- ultralytics (YOLO v11)
- opencv-python
- numpy
- pillow
- torch (PyTorch backend)

## Usage Examples
```bash
# Basic pose matching
python3 main.py --target data/target_images/image.jpg --comparison-dir data/comparison_images

# With visualization (masking and skeleton on by default)
python3 main.py --target data/target_images/image.jpg --comparison-dir data/comparison_images --visualize

# Control visualization elements
python3 main.py --target data/target_images/image.jpg --comparison-dir data/comparison_images --visualize --no-skeleton --no-mask
```
