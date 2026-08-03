# Poser: Skeletal & Pose-Driven Video Art Synthesis Engine

**`poser`** is an intelligent Python system designed to synthesize video art by extracting skeletal human pose trajectories from target video clips (e.g. a 15-second dance scene) and reconstructing them frame-by-frame using closely matching poses retrieved from an indexed library of source films (e.g., hundreds of Hollywood movies).

---

## Concept & Artistic Vision

Imagine taking a 15-second dance sequence or martial arts routine and watching it play out in real-time, where **every single frame comes from a completely different film**, seamless in pose alignment but wildly diverse in visual style, lighting, color, and character.

```
[ Target Clip (15s Dance) ] ──> Extract Keypoint Trajectory
                                          │
                                          ▼
                      [ High-Performance Binary Pose Database ]
                           (Ingested 100+ Hollywood Films)
                                          │
                                          ▼
                      [ Diversity & Coarse-to-Fine Search ]
                     - Multi-film constraint (no repeated films)
                     - Interval lookup + local ±2s window search
                                          │
                                          ▼
                      [ Synthesized Composite Video Art ]
```

---

## Features

- **Massive Media Ingestion**: Single-pass pose extraction per movie with YOLOv11/v13.
- **Ultra-Compact Pose Database**: Replaces JSON logs with compact SQLite binary tables storing 34D/51D normalized float16/float32 pose BLOBs.
- **Coarse-to-Fine Vector Matching**:
  - **Interval Sampling**: Fast keyframe pose lookup across millions of frames.
  - **Temporal Refinement Window**: Inspects a $\pm 2.0$-second local window around matching candidates to assess temporal velocity and motion continuity.
- **Diversity & Cooldown Rules**:
  - `--exclude-same-film`: Disallows matching frames from the same source movie as target or prior frames.
  - `--max-clips-per-film`: Limits maximum frame contributions per source film.
  - `--film-cooldown`: Forces a minimum number of frames before a source film can be reused.
- **LLM Agent & Skill Native**: Built-in agent configurations and skills for **Gemini 3** (`.gemini/skills`), **Claude Code** (`CLAUDE.md`, `.claude/skills`), and **OpenCode** (`AGENTS.md`, `.opencode/skills`).
- **Diagnostic Visualizer & Video Composite Engine**: Generates skeletal overlays, diagnostic comparison grids, body segmentation masking, and direct FFmpeg video assembly.

---

## Database Architecture: High Efficiency Pose Storage

Storing keypoint data for 100 movies (approx. 17,000,000 frames) in standard JSON format would take over **15 GB** of disk space and saturate RAM. `poser` solves this using an optimized binary database format:

- **Keypoint Compression**: 17 COCO keypoints $(x, y, c)$ normalized and packed into a raw 34D/51D `float16` binary BLOB ($68$ to $102$ bytes per pose).
- **Indexing**: SQLite database with composite indexing on `(film_id, frame_idx, timestamp)`.
- **Memory Footprint**: 17 million frames fit in under **1.2 GB** of disk space and query in milliseconds.

---

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/poser.git
cd poser
```

2. Create and activate a Python virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Usage & Workflows

### 1. Ingest Video Media Library
Extract and store skeletal poses from a directory of source movies:
```bash
python3 main.py ingest --input-dir /path/to/movies_library --db pose_library.db --fps 12
```

### 2. Reconstruct Video Art Sequence
Reconstruct a 15-second target clip using the ingested pose library:
```bash
python3 main.py reconstruct \
  --target input_dance_clip.mp4 \
  --db pose_library.db \
  --output reconstructed_art.mp4 \
  --exclude-same-film \
  --max-clip-reuse 2 \
  --temporal-window 2.0 \
  --visualize
```

### 3. Database Statistics
Inspect total ingested films, frame counts, and storage efficiency:
```bash
python3 main.py db-stats --db pose_library.db
```

---

## LLM Agent & Skill Files

This project includes agent guidelines and skills for modern LLM tools:

| Assistant / CLI | File Paths |
| :--- | :--- |
| **Gemini 3 / AGY** | [`GEMINI.md`](file:///Users/grayson/workbench/poser/GEMINI.md), [`.gemini/skills/pose-video-art/SKILL.md`](file:///Users/grayson/workbench/poser/.gemini/skills/pose-video-art/SKILL.md) |
| **Claude Code** | [`CLAUDE.md`](file:///Users/grayson/workbench/poser/CLAUDE.md), [`.claude/skills/video-art-matcher/SKILL.md`](file:///Users/grayson/workbench/poser/.claude/skills/video-art-matcher/SKILL.md) |
| **OpenCode** | [`AGENTS.md`](file:///Users/grayson/workbench/poser/AGENTS.md), [`.opencode/skills/video-art-reconstruction/SKILL.md`](file:///Users/grayson/workbench/poser/.opencode/skills/video-art-reconstruction/SKILL.md) |

---

## Testing

Run automated tests using pytest:
```bash
python3 -m pytest tests/ -v
```

---

## License

MIT License. See LICENSE file for details.
