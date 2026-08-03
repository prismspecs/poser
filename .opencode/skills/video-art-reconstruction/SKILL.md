---
name: video-art-reconstruction
description: OpenCode skill for pose data ingestion, fast vector index search, and diversity-constrained frame retrieval.
---

# Video Art Reconstruction Skill for OpenCode

## Workflow
1. **Pose Extraction**: Extract human skeletons from video media using YOLO pose detection.
2. **Binary Database Storage**: Store pose features in binary SQLite tables with indexed normalized vectors.
3. **Sequence Reconstruction**: Match a 15-second target clip frame-by-frame against the database using vector distance (MSE/Cosine) and temporal refinement window ($\pm 2.0\text{s}$).
4. **Diversity Enforcement**: Apply source exclusion flags and clip reuse quotas to maximize visual variation across films.
