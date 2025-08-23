# YOLO v13 Pose Estimation Project

A Python application that uses YOLO v13 for pose estimation to find the closest pose match between a target image and multiple comparison images.

## Features

- **YOLO v13 Integration**: State-of-the-art pose estimation using the latest YOLO model
- **Pose Matching**: Advanced algorithms to find similar poses across multiple images
- **Multiple Distance Metrics**: Support for Euclidean, Manhattan, and cosine similarity
- **Confidence Weighting**: Pose matching weighted by detection confidence
- **Batch Processing**: Process multiple comparison images efficiently
- **Comprehensive Output**: Detailed similarity scores and keypoint analysis
- **Visualization**: Built-in pose visualization capabilities

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for optimal performance)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd poser
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install PyTorch with CUDA support for GPU acceleration:
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Usage

### Basic Usage

Find the closest pose match between a target image and comparison images:

```bash
python main.py --target target.jpg --comparison-dir ./comparison_images/
```

### Advanced Options

```bash
python main.py \
    --target target.jpg \
    --comparison-dir ./comparison_images/ \
    --threshold 0.8 \
    --max-results 20 \
    --output results.json \
    --verbose
```

### Command Line Arguments

- `--target`: Path to the target image for pose matching (required)
- `--comparison-dir`: Directory containing comparison images (required)
- `--threshold`: Confidence threshold for pose detection (default: 0.5)
- `--max-results`: Maximum number of results to return (default: 10)
- `--output`: Output file for results in JSON format (optional)
- `--verbose`: Enable verbose output (optional)

### Example Output

```
Found 5 pose matches:
--------------------------------------------------------------------------------
 1. person_standing.jpg
    Similarity Score: 0.892
    Rank: 1

 2. person_walking.jpg
    Similarity Score: 0.756
    Rank: 2

 3. person_sitting.jpg
    Similarity Score: 0.634
    Rank: 3
```

## Project Structure

```
poser/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── main.py                  # Main application entry point
├── pose_estimator.py        # YOLO pose estimation wrapper
├── pose_matcher.py          # Pose similarity matching
├── utils/                   # Utility modules
│   ├── __init__.py
│   ├── image_utils.py       # Image loading and processing
│   └── pose_utils.py        # Pose data structures and utilities
├── models/                  # Model-related modules
│   └── __init__.py
├── data/                    # Data directories
│   ├── target_images/       # Target images for matching
│   └── comparison_images/   # Images to compare against
├── results/                 # Output results
└── tests/                   # Test modules
    ├── __init__.py
    └── test_pose_estimator.py
```

## API Reference

### PoseEstimator

Main class for YOLO pose estimation:

```python
from pose_estimator import PoseEstimator

# Initialize with custom confidence threshold
estimator = PoseEstimator(confidence_threshold=0.7, model_size="s")

# Extract poses from image
poses = estimator.extract_poses(image, image_path)

# Get model information
info = estimator.get_model_info()
```

### PoseMatcher

Class for pose similarity matching:

```python
from pose_matcher import PoseMatcher

# Initialize matcher
matcher = PoseMatcher(distance_metric="euclidean", normalize_keypoints=True)

# Find best match
best_match = matcher.find_best_match(target_pose, comparison_poses)

# Rank all poses
ranked_results = matcher.rank_poses(target_pose, comparison_poses)
```

### Data Structures

#### PoseData

```python
@dataclass
class PoseData:
    keypoints: List[Optional[Tuple[float, float, float]]]  # x, y, confidence
    bounding_box: Tuple[float, float, float, float]        # x1, y1, x2, y2
    confidence_score: float
    image_path: str
    pose_id: str
```

#### SimilarityResult

```python
@dataclass
class SimilarityResult:
    target_image: str
    comparison_image: str
    similarity_score: float
    keypoint_distances: List[float]
    rank: int
```

## Configuration

### Model Sizes

Available YOLO model sizes:
- `n` (nano): Fastest, lowest accuracy
- `s` (small): Good balance of speed and accuracy
- `m` (medium): Higher accuracy, slower
- `l` (large): High accuracy, slower
- `x` (xlarge): Highest accuracy, slowest

### Distance Metrics

- **Euclidean**: Standard Euclidean distance between keypoints
- **Manhattan**: L1 distance (sum of absolute differences)
- **Cosine**: Cosine similarity between keypoint vectors

## Performance

- **GPU Acceleration**: 10-50x faster with CUDA-compatible GPU
- **Batch Processing**: Efficient processing of multiple images
- **Memory Management**: Optimized for large image collections
- **Caching**: Automatic model caching for repeated use

## Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_pose_estimator.py

# Run with coverage
python -m pytest --cov=. tests/
```

## Examples

### Basic Pose Matching

```python
from pose_estimator import PoseEstimator
from pose_matcher import PoseMatcher
from utils.image_utils import load_image

# Load images
target_image = load_image("target.jpg")
comparison_image = load_image("comparison.jpg")

# Extract poses
estimator = PoseEstimator()
target_poses = estimator.extract_poses(target_image, "target.jpg")
comparison_poses = estimator.extract_poses(comparison_image, "comparison.jpg")

# Find matches
matcher = PoseMatcher()
if target_poses and comparison_poses:
    best_match = matcher.find_best_match(target_poses[0], comparison_poses)
    print(f"Similarity Score: {best_match.similarity_score:.3f}")
```

### Custom Similarity Calculation

```python
# Use different distance metric
matcher = PoseMatcher(distance_metric="cosine", normalize_keypoints=False)

# Get detailed statistics
stats = matcher.get_matching_statistics(target_poses[0], comparison_poses)
print(f"Average similarity: {stats['avg_similarity']:.3f}")
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce model size or batch size
2. **No Poses Detected**: Lower confidence threshold
3. **Slow Performance**: Use GPU acceleration or smaller model
4. **Import Errors**: Ensure all dependencies are installed

### Performance Tips

- Use GPU acceleration when available
- Choose appropriate model size for your use case
- Process images in batches for efficiency
- Use appropriate confidence thresholds

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO v13 implementation
- [COCO Dataset](https://cocodataset.org/) for pose keypoint definitions
- [OpenCV](https://opencv.org/) for computer vision utilities

## Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review the test examples

---

**Note**: This project requires significant computational resources for optimal performance. Consider using GPU acceleration for production use cases.
