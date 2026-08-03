---
name: pose-video-art
description: Skill for extracting skeletal pose data from source videos, building binary lookup indices, and reconstructing target video clips with diversity constraints.
---

# Pose-Driven Video Art Synthesis Skill

This skill provides step-by-step instructions for operating, extending, and debugging the `poser` video art synthesis pipeline.

## 1. Video Ingestion & Binary Indexing
When ingesting new source videos (e.g. movies, dance clips):
- Process videos using single-pass YOLO pose estimation.
- Store normalized keypoint coordinates (17 COCO keypoints: x, y, confidence) as packed binary BLOBs in SQLite or HDF5.
- Key schema: `film_id`, `frame_idx`, `timestamp`, `pose_vector_34d`, `bbox`, `confidence`.

## 2. Coarse-to-Fine Matching & Diversity Constraints
To reconstruct a target video clip (e.g., 15 seconds) from a source library:
- **Interval Sampling**: Query keyframe poses every $N$ frames (e.g., every 6-12 frames).
- **Temporal Refinement Window**: For selected candidate frames, inspect a local window ($\pm 2.0$ seconds) to evaluate temporal motion continuity and keypoint velocity.
- **Diversity Flags**:
  - `exclude_same_film`: Prevent picking frames from the source film used by the target or previous frame.
  - `max_clips_per_film`: Limit cumulative frames from a single source video.
  - `film_switch_penalty`: Balance smooth temporal continuation vs maximum visual diversity.

## 3. Execution Commands
```bash
# Ingest dataset
python3 main.py ingest --input-dir data/source_movies --db data/poses.db

# Reconstruct video art
python3 main.py reconstruct --target target_sequence.mp4 --db data/poses.db --output art_render.mp4 --exclude-same-film
```
