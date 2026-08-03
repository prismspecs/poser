# Poser: Skeletal & Pose-Driven Video Art Synthesis

`poser` is a high-performance Python system designed to create composite video art by matching skeletal/pose trajectories from a target video clip against a massive indexed library of source films (e.g., Hollywood movie datasets).

---

## 🚀 Architectural Goals & Workflows

1. **Massive Video Ingestion & Compact Database Indexing**:
   - Single-pass pose extraction per movie with compact binary storage (SQLite + Compressed NumPy vectors or HDF5/Parquet).
   - High-density or interval sampling (e.g., keyframes + local window fine-search).
   - Instant vector-similarity lookup (Cosine / L2 distance on 34D normalized COCO keypoint vectors).

2. **Sequence Reconstruction & Diversity Constraints**:
   - Frame-by-frame or clip-by-clip pose sequence matching.
   - **Diversity Controls**: `--exclude-same-film`, `--max-clips-per-film`, `--min-film-switch-cooldown`, `--diversity-bias`.
   - **Temporal Continuity**: Multi-frame trajectory matching (velocity & motion smoothness) to balance diversity with visual flow.

3. **High-Quality Video Synthesis & Visualization**:
   - Frame retrieval, pose alignment overlay, body masking, and FFmpeg video assembly.

---

## 🛠 Command Reference

### Database & Ingestion
```bash
# Ingest video library into optimized pose database
python3 main.py ingest --input-dir /path/to/movies --db pose_library.db --fps 12 --batch-size 64

# Inspect database stats (total films, frames, poses, storage size)
python3 main.py db-stats --db pose_library.db
```

### Video Art Reconstruction
```bash
# Reconstruct target video using source video library with diversity constraints
python3 main.py reconstruct \
  --target input_dance.mp4 \
  --db pose_library.db \
  --output output_art.mp4 \
  --exclude-same-film \
  --max-clip-reuse 3 \
  --temporal-window 5 \
  --visualize
```

### Testing & Validation
```bash
# Run pytest suite
python3 -m pytest tests/ -v

# Run pose matching verification script
python3 test_random_poses.py
```

---

## 📁 Repository Structure

```
poser/
├── main.py              # CLI entry point (ingest, reconstruct, db-stats)
├── pose_estimator.py    # YOLO-based pose detection & keypoint extraction
├── pose_matcher.py      # Pose normalization, vector similarity, diversity constraints
├── pose_cache.py        # Database storage adapter (SQLite/Binary/JSON fallback)
├── pose_visualizer.py   # Overlay, mask, and diagnostic rendering
├── utils/
│   ├── image_utils.py   # Frame extraction & OpenCV operations
│   └── pose_utils.py    # PoseData structures & spatial normalization
├── tests/               # Unit and integration test suite
├── CLAUDE.md            # Claude Code instructions & architecture reference
├── GEMINI.md            # Gemini / Antigravity instructions
├── AGENTS.md            # OpenCode & multi-agent system prompt
└── plan.md              # Detailed technical specification & roadmap
```

---

## 💡 Developer & Coding Guidelines

- **Pose Vector Standard**: 17 COCO keypoints (x, y, confidence). Poses must be normalized relative to torso center (hip/shoulder midpoint) and torso scale for orientation- and scale-invariant matching.
- **Storage Efficiency**: Never store raw JSON for large-scale datasets. Use binary packed arrays (float16/float32) in SQLite BLOBs or HDF5/Parquet.
- **Error Handling**: Gracefully handle missing keypoints or low-confidence detections using adaptive fallback thresholds.
- **Code Style**: Standard Python PEP8, type hints (`typing.Optional`, `Tuple`, `List`, `NDArray`), modular decoupled functions.
