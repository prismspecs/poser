# Poser: Technical Blueprint & Implementation Roadmap

## 🎯 Architecture Overview

```
[ Input Target Video ]
         │
         ▼
[ Pose Estimator (YOLOv11/v13) ] ──> Normalized 34D Keypoint Vector Sequence
                                                  │
                                                  ▼
                                     [ Coarse-to-Fine Search ]
                                                  │
                      ┌───────────────────────────┴───────────────────────────┐
                      ▼                                                       ▼
        [ Interval Keyframe Lookup ]                           [ Temporal Window ±2.0s Search ]
        Fast KNN search across index                            Fine motion velocity matching
                      │                                                       │
                      └───────────────────────────┬───────────────────────────┘
                                                  │
                                                  ▼
                                   [ Diversity Constraint Filter ]
                                  - Source Film Exclusions
                                  - Clip Reuse Caps & Cooldowns
                                                  │
                                                  ▼
                                  [ Frame Retrieval & Renderer ]
                                  FFmpeg Video Compositing & Overlay
```

---

## 🗄 1. Compact Binary Database Schema (SQLite / NumPy BLOBs)

### Problem with JSON Cache
`pose_cache.json` currently stores keypoints as verbose JSON strings, resulting in ~15 MB for a few frames. For 100 films (~17M frames), JSON would require over 15-20 GB and create severe I/O bottlenecks.

### SQLite Binary Table Schema
```sql
CREATE TABLE IF NOT EXISTS films (
    film_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT UNIQUE,
    filepath TEXT,
    total_frames INTEGER,
    fps REAL,
    duration REAL
);

CREATE TABLE IF NOT EXISTS poses (
    pose_id INTEGER PRIMARY KEY AUTOINCREMENT,
    film_id INTEGER,
    frame_idx INTEGER,
    timestamp REAL,
    bbox_x1 REAL, bbox_y1 REAL, bbox_x2 REAL, bbox_y2 REAL,
    confidence REAL,
    -- 34D float16 binary BLOB (17 pairs of normalized x, y coordinates = 68 bytes)
    normalized_vector BLOB,
    FOREIGN KEY(film_id) REFERENCES films(film_id)
);

CREATE INDEX IF NOT EXISTS idx_film_frame ON poses(film_id, frame_idx);
```

---

## 🔍 2. Coarse-to-Fine Matching & Local Window Refinement

### Interval Sampling & Keyframe Indexing
1. **Coarse Search**: Store or index keyframes at sampled intervals (e.g. every 6-12 frames / 0.25-0.5s). Compute candidate matches using fast vector distance (MSE / Cosine similarity) on 34D normalized pose vectors.
2. **Fine Search (Local Window Refinement)**:
   - When a top candidate pose is identified at frame $T_{\text{candidate}}$ of `Film_X`, evaluate a local temporal window $[T_{\text{candidate}} - 2.0\text{s}, T_{\text{candidate}} + 2.0\text{s}]$ (approx. $\pm 24-48$ frames).
   - Compute keypoint velocity vectors $(\Delta x, \Delta y)$ to match motion direction and prevent abrupt, jerky jumps.

---

## 🎨 3. Diversity Constraints & Film Switch Heuristics

To ensure maximum visual variety without sacrificing motion quality:

- **`exclude_same_film`**: Flag to prohibit selecting frames from the film used by the target sequence or current segment.
- **`max_clips_per_film`**: Hard ceiling on the total number of frames pulled from any single source video.
- **`film_cooldown_frames`**: Number of frames that must elapse before a previously selected film can be chosen again.
- **`film_switch_penalty`**: Weight factor added to distance score to penalize overly rapid film switching when temporal continuity is preferred.

---

## 🤖 4. Integrated AI Tooling Standards

- **Gemini 3**: [`GEMINI.md`](file:///Users/grayson/workbench/poser/GEMINI.md), [`.gemini/skills/pose-video-art/SKILL.md`](file:///Users/grayson/workbench/poser/.gemini/skills/pose-video-art/SKILL.md)
- **Claude Code**: [`CLAUDE.md`](file:///Users/grayson/workbench/poser/CLAUDE.md), [`.claude/skills/video-art-matcher/SKILL.md`](file:///Users/grayson/workbench/poser/.claude/skills/video-art-matcher/SKILL.md)
- **OpenCode**: [`AGENTS.md`](file:///Users/grayson/workbench/poser/AGENTS.md), [`.opencode/skills/video-art-reconstruction/SKILL.md`](file:///Users/grayson/workbench/poser/.opencode/skills/video-art-reconstruction/SKILL.md)
