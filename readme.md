# Pose Estimation and Matching System

A Python-based system for estimating human poses from images using YOLO v13 and finding similar poses across a dataset.

## Features

- **Pose Estimation**: Extract human poses from images using YOLO v13
- **Pose Matching**: Find similar poses using advanced similarity algorithms
- **Multi-Person Detection**: Handle images with multiple people
- **Visualization**: Generate diagnostic images showing poses and similarity scores
- **CLI Interface**: Easy-to-use command-line interface

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd poser
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

Compare a target image against a directory of comparison images:

```bash
python3 main.py --target path/to/target.jpg --comparison-dir path/to/comparison/images
```

### Generate Visualizations

Add the `--visualize` flag to create diagnostic images:

```bash
python3 main.py --target path/to/target.jpg --comparison-dir path/to/comparison/images --visualize
```

This will create:
- `pose_comparison_[target].jpg` - Main grid showing target, overlay, and comparisons
- `keypoint_analysis_[target].jpg` - Detailed keypoint comparison
- `pose_overlay_[target].jpg` - Target with winning pose skeleton overlaid

### Testing with Sample Images

Use the included test script to randomly test poses:

```bash
python3 test_random_poses.py
```

## Usage Examples

### Command Line Options

```bash
python3 main.py --help
```

Available options:
- `--target`: Path to target image
- `--comparison-dir`: Directory containing comparison images
- `--threshold`: Confidence threshold for pose detection (default: 0.5)
- `--max-results`: Maximum number of results to return (default: 10)
- `--output`: Save results to JSON file
- `--visualize`: Generate diagnostic visualizations
- `--output-dir`: Directory for visualization outputs (default: "results")
- `--verbose`: Enable verbose output

### Example Commands

```bash
# Basic pose matching
python3 main.py --target data/test_images/basketball1.jpg --comparison-dir data/test_images

# With visualization and custom output
python3 main.py --target data/test_images/basketball1.jpg --comparison-dir data/test_images --visualize --output-dir my_results --max-results 5

# Save results to file
python3 main.py --target data/test_images/basketball1.jpg --comparison-dir data/test_images --output results.json
```

## Architecture

### Core Components

- **`main.py`**: Main CLI application and orchestration
- **`pose_estimator.py`**: YOLO-based pose estimation
- **`pose_matcher.py`**: Pose similarity calculation and matching
- **`pose_visualizer.py`**: Diagnostic visualization generation
- **`utils/`**: Utility functions for image processing and pose data

### Data Structures

- **`PoseData`**: Contains keypoints, bounding box, confidence, and metadata
- **`SimilarityResult`**: Stores similarity scores and comparison information

### Pose Similarity Algorithm

The system uses a sophisticated pose comparison algorithm:

1. **Normalization**: Makes poses position, scale, and orientation invariant
2. **Keypoint Alignment**: Uses torso (shoulders/hips) as reference for alignment
3. **Distance Calculation**: Computes Mean Squared Error of normalized keypoint distances
4. **Similarity Scoring**: Converts distances to similarity scores using exponential decay

## Testing

### Running Tests

```bash
# Run all tests
python3 -m pytest tests/

# Run specific test file
python3 -m pytest tests/test_pose_estimator.py

# Run with verbose output
python3 -m pytest -v tests/
```

### Test Images

The system includes sample test images in `data/test_images/`:
- Basketball poses (basketball1.jpg, basketball2.jpg, basketball3.jpg)
- Dance poses (dancing1.jpg, dancing2.jpg, dancing3.jpg, dancing4.jpg)

## Output and Results

### Similarity Scores

Similarity scores range from 0.0 to 1.0:
- **0.8-1.0**: Very similar poses
- **0.6-0.8**: Similar poses
- **0.4-0.6**: Moderately similar poses
- **0.2-0.4**: Somewhat similar poses
- **0.0-0.2**: Different poses

### Visualization Outputs

When using `--visualize`, the system generates:

1. **Main Comparison Grid** (`pose_comparison_[target].jpg`):
   - Target image with skeleton (top, centered)
   - Overlay image showing target + winning pose skeleton
   - All comparison images with their best matching pose skeletons

2. **Keypoint Analysis** (`keypoint_analysis_[target].jpg`):
   - Detailed comparison of individual keypoints
   - Distance analysis between corresponding body parts

3. **Pose Overlay** (`pose_overlay_[target].jpg`):
   - Target image with winning pose skeleton overlaid
   - Color-coded: Blue (target), Orange (winning pose)

## Performance and Optimization

### Multi-Person Handling

- Automatically detects multiple people in images
- Selects highest confidence pose as target
- Compares target against ALL people in comparison images
- Uses best matching pose from each comparison image

### Memory Management

- Efficient image loading and processing
- Optimized pose data structures
- Minimal memory footprint during processing

## Troubleshooting

### Common Issues

1. **No poses detected**: Lower the confidence threshold with `--threshold 0.3`
2. **Poor similarity scores**: Check image quality and pose clarity
3. **Visualization errors**: Ensure output directory exists and has write permissions

### Debug Mode

Use `--verbose` for detailed logging:

```bash
python3 main.py --target image.jpg --comparison-dir images/ --verbose
```

## Project Status

### Completed Features ✅

- **Core Infrastructure**: Project structure, YOLO integration, pose estimation pipeline
- **Pose Matching**: Keypoint extraction, normalization, similarity algorithms, ranking
- **User Interface**: CLI interface, batch processing, results visualization
- **Optimization**: Performance improvements, memory management, error handling
- **Multi-Person Detection**: Handles multiple humans in images
- **Advanced Visualization**: Diagnostic images with pose overlays and skeleton drawing

### Current Capabilities

- **Pose Detection**: YOLO v13 with configurable confidence thresholds
- **Similarity Calculation**: Normalized pose comparison with MSE-based scoring
- **Visualization**: Comprehensive diagnostic suite with pose overlays
- **CLI Interface**: Full-featured command-line application
- **Testing**: Automated testing with sample images

### Technical Details

- **Model**: YOLO v13 pose estimation (falls back to YOLOv8n-pose.pt)
- **Keypoints**: 17 COCO format keypoints per person
- **Similarity**: Position, scale, and orientation invariant comparison
- **Performance**: Optimized for real-time processing and batch operations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

[Add your license information here]

## Acknowledgments

- YOLO v13 for pose estimation
- OpenCV for image processing
- PyTorch for deep learning backend
