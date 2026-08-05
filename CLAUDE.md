# Poser: Skeletal & Pose-Driven Video Art Synthesis

`poser` is a high-performance Python system designed to create composite video art by matching skeletal/pose trajectories from a target video clip against a massive indexed library of source films (e.g., Hollywood movie datasets).

---

## Architectural Goals & Workflows

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

## Command Reference

### Database & Ingestion
```bash
# Ingest video library into optimized pose database
python3 main.py ingest --input-dir /path/to/movies --db pose_library.db --fps 12 --batch-size 64

# Inspect database stats (total films, frames, poses, storage size)
python3 main.py db-stats --db pose_library.db

# Drop poses whose source frame contains no verifiable person
python3 clean_db.py --db pose_library.db

# Repair frame->timestamp mappings in a database ingested before the
# natural-sort fix (dry run by default; no-op for films under 9999 frames)
python3 repair_db_timestamps.py --db pose_library.db --apply
```

**Frame indexing**: `frame_idx` is a position in the resampled `--fps` stream,
*not* a native video frame number. Always retrieve source frames with
`extract_frame_at_timestamp(path, timestamp)`. Seeking with `CAP_PROP_POS_FRAMES`
on `frame_idx` silently reads the wrong moment. List extracted frames with
`natural_frame_key`, since `frame_%04d` widens past 9999 and plain sorting
scrambles the ordering.

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

## Repository Structure

```
poser/
├── main.py                    # CLI entry point (ingest, reconstruct, db-stats)
├── pose_estimator.py          # YOLO-based pose detection & keypoint extraction
├── pose_matcher.py            # Vector similarity & diversity constraints
├── pose_db.py                 # SQLite pose library (binary float16 vectors)
├── pose_cache.py              # On-disk cache of per-image pose detections
├── pose_visualizer.py         # Cutout compositing, overlay & diagnostic rendering
├── clean_db.py                # Prune poses with no verifiable person
├── repair_db_timestamps.py    # Fix frame->timestamp maps from pre-fix ingests
├── test_pipeline.py           # Standalone per-stage diagnostics (local paths)
├── test_random_poses.py       # Pose matching spot-check script
├── utils/
│   ├── image_utils.py         # Frame extraction, timestamp seeking, natural sort
│   └── pose_utils.py          # PoseData structures & spatial normalization
├── tests/                     # pytest suite
├── test-vids/input.mkv        # Standing test target for `reconstruct`
├── results/                   # Generated renders (gitignored)
├── CLAUDE.md                  # Claude Code instructions & architecture reference
├── GEMINI.md                  # Gemini / Antigravity instructions
├── AGENTS.md                  # OpenCode & multi-agent system prompt
└── plan.md                    # Detailed technical specification & roadmap
```

`pose_db.py` is the pose library; `pose_cache.py` is an unrelated detection
cache keyed by image identity. Video-derived frames are cached under
`"<film path>@<timestamp>"` rather than a real file path.

---

## Developer & Coding Guidelines

- **Pose Vector Standard**: 17 COCO keypoints (x, y, confidence). Poses must be normalized relative to torso center (hip/shoulder midpoint) and torso scale for orientation- and scale-invariant matching.
- **Storage Efficiency**: Never store raw JSON for large-scale datasets. Use binary packed arrays (float16/float32) in SQLite BLOBs or HDF5/Parquet.
- **Error Handling**: Gracefully handle missing keypoints or low-confidence detections using adaptive fallback thresholds.
- **Code Style**: Standard Python PEP8, type hints (`typing.Optional`, `Tuple`, `List`, `NDArray`), modular decoupled functions.
