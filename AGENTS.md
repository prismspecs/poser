# AGENTS.md - OpenCode & Multi-Agent Architecture for Poser

## Overview
`poser` creates generative video art by re-assembling video frames from a large ingested dataset (e.g., Hollywood films) based on skeletal pose matching against a target video clip.

## Subagents & Roles

1. **Ingestion & Indexing Agent**:
   - Manages video decoding via OpenCV / PyAV.
   - Runs pose estimation using YOLOv11/v13.
   - Serializes normalized keypoint vectors into binary SQLite / HDF5 / Parquet database tables.

2. **Pose Matcher & Diversity Agent**:
   - Implements normalized pose similarity math (torso alignment, keypoint MSE).
   - Solves non-repetitive sequence matching using diversity constraints (`exclude_same_film`, `max_clip_reuse`).
   - Implements two-stage coarse-to-fine temporal search (+/- 2 second window search).

3. **Rendering & Compositing Agent**:
   - Re-assembles winning frames into composite output video using FFmpeg.
   - Renders skeletal diagnostic overlays and background masks.

## Key Instructions for Agents
- When extending the pose database, preserve backwards compatibility with SQLite BLOB keypoint arrays.
- Avoid loading full JSON caches into memory for datasets exceeding 10,000 frames.
- Test changes using `python3 -m pytest tests/`.
