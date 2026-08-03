---
name: video-art-matcher
description: Claude Code skill for running and developing the pose estimation, binary index lookup, and video art reconstruction system.
---

# Video Art Matcher Skill for Claude Code

Use this skill when developing features, running pose estimation, or tuning similarity heuristics for `poser`.

## Key Architectural Principles

1. **Normalized Keypoint Representation**:
   - Translate all 17 COCO keypoints relative to torso midpoint.
   - Scale keypoints by distance between neck/shoulders and hips.
   - Store normalized vectors as binary NumPy arrays (`float16` or `float32`) in SQLite or HDF5.

2. **Diversity & Sequence Matching**:
   - Optimize target sequence frame matching via dynamic programming or greedy nearest-neighbor with penalty terms.
   - Implement source film cooldown and clip reuse caps.

3. **Verification**:
   - Verify performance and keypoint math with unit tests: `python3 -m pytest tests/`.
