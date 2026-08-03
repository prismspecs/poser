# Gemini System Prompt & Guidelines: Poser Repository

This repository implements **Poser**, a skeletal pose extraction, vector index lookup, and video art reconstruction pipeline.

## Model Configuration & Rules
- Always use Gemini 3 models (`gemini-3-flash`, `gemini-3-pro`) for code analysis, architecture design, and subagent orchestration.

## Core Capabilities & Objectives

1. **Optimized Binary Storage Engine**:
   - High performance SQLite / HDF5 / Parquet vector store for storing millions of extracted human poses from movie datasets.
   - 34D/51D normalized pose vectors for fast nearest-neighbor matching using vector similarity (MSE, Cosine, FAISS/USearch indexing).

2. **Temporal & Diversity-Constrained Matcher**:
   - Matching a target video clip (e.g., 15s sequence) against source library frames.
   - Enforce film diversity rules (`--exclude-same-film`, `--max-clips-per-film`, `--cooldown-frames`).
   - Coarse-to-Fine search (interval keyframes + local +/- 2s temporal window refinement).

3. **CLI & Rendering Engine**:
   - Command-line workflows for `ingest`, `reconstruct`, and `visualize`.
   - FFmpeg integration with frame extraction, overlay, and video encoding.

## Skills Directory
- Gemini & Antigravity skills are defined in `.gemini/skills/pose-video-art/SKILL.md`.
