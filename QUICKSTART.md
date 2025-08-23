# Quick Start Guide

This guide will help you get up and running with the YOLO v13 Pose Estimation project in minutes.

## Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)

## Setup

1. **Clone and navigate to the project:**
   ```bash
   cd poser
   ```

2. **Create and activate virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Quick Test

1. **Run the demo to verify everything works:**
   ```bash
   python demo.py
   ```

2. **Run tests to ensure functionality:**
   ```bash
   python -m pytest tests/ -v
   ```

## Basic Usage

### 1. Prepare Your Images

- **Target Image**: The pose you want to find matches for
- **Comparison Images**: A folder of images to search through

### 2. Run Pose Matching

```bash
python main.py --target target.jpg --comparison-dir ./comparison_images/ --verbose
```

### 3. Advanced Options

```bash
python main.py \
    --target target.jpg \
    --comparison-dir ./comparison_images/ \
    --threshold 0.8 \
    --max-results 20 \
    --output results.json \
    --verbose
```

## Example Workflow

1. **Place your target image** in the project directory
2. **Create a comparison folder** with images to search
3. **Run the matching:**
   ```bash
   python main.py --target my_target.jpg --comparison-dir ./my_images/ --verbose
   ```
4. **View results** in the terminal or saved JSON file

## Sample Data Structure

```
poser/
├── target.jpg                    # Your target image
├── comparison_images/            # Folder with images to search
│   ├── person1.jpg
│   ├── person2.jpg
│   └── person3.jpg
└── results.json                  # Output results (optional)
```

## Troubleshooting

- **No poses detected**: Lower the confidence threshold (`--threshold 0.3`)
- **Slow performance**: Use smaller model size or GPU acceleration
- **Import errors**: Ensure virtual environment is activated

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check the [API Reference](README.md#api-reference) for advanced usage
- Explore the [demo.py](demo.py) script for examples

## Support

- Check the [README.md](README.md) for comprehensive documentation
- Review error messages for troubleshooting hints
- Ensure all dependencies are properly installed
