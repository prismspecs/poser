"""
Repair frame_idx -> timestamp mappings corrupted by lexicographic frame sorting.

Ingest indexed poses by their position in ``sorted(glob("frame_*.jpg"))``. Because
ffmpeg's ``frame_%04d`` pattern widens past four digits, that sort interleaves
``frame_10000.jpg`` right after ``frame_1000.jpg`` for any film yielding more
than 9999 sampled frames, so the stored timestamps for those films point at the
wrong moment. This script inverts the permutation and rewrites the timestamps.

Films with 9999 or fewer sampled frames are unaffected; for them the repair is a
no-op, which doubles as a correctness check on the inversion.

Usage:
    python3 repair_db_timestamps.py --db pose_library.db [--apply] [--verify]
"""

import argparse
import sqlite3
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def probe_duration(video_path: str) -> Optional[float]:
    """Return a video's duration in seconds via ffprobe, or None on failure."""
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True, text=True, timeout=60,
        )
        return float(out.stdout.strip()) if out.returncode == 0 else None
    except Exception:
        return None


def sorted_position_to_frame_number(total_frames: int) -> List[int]:
    """
    Rebuild the mapping ingest used: sorted position -> real frame number.

    Args:
        total_frames: Number of ``frame_%04d.jpg`` files ffmpeg produced.

    Returns:
        List where index i holds the 1-based frame number that sat at sorted
        position i during ingest.
    """
    names = [(f"frame_{i:04d}.jpg", i) for i in range(1, total_frames + 1)]
    names.sort(key=lambda pair: pair[0])
    return [num for _, num in names]


def score_mapping(
    conn: sqlite3.Connection, film_id: int, mapping: List[int], fps: float, samples: int
) -> float:
    """
    Score a candidate mapping by how monotonic the recovered timestamps are.

    Poses are ingested in scan order, so pose_id and true time must rise
    together. A wrong total_frames shuffles that relationship, making the
    fraction of ascending consecutive pairs a cheap, model-free quality signal.

    Args:
        conn: Open database connection.
        film_id: Film to score.
        mapping: Candidate sorted-position -> frame-number table.
        fps: Ingest sampling rate.
        samples: Maximum poses to sample.

    Returns:
        Fraction of consecutive sampled poses whose recovered time increases.
    """
    rows = conn.execute(
        "SELECT frame_idx FROM poses WHERE film_id=? ORDER BY pose_id LIMIT ?",
        (film_id, samples),
    ).fetchall()
    times = [
        (mapping[r[0]] - 1) / fps
        for r in rows
        if 0 <= r[0] < len(mapping)
    ]
    if len(times) < 2:
        return 0.0
    ascending = sum(1 for a, b in zip(times, times[1:]) if b > a)
    return ascending / (len(times) - 1)


def repair(db_path: str, apply: bool, samples: int = 400) -> None:
    """Recompute timestamps for every film in the database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    films = conn.execute(
        "SELECT film_id, title, filepath, fps FROM films"
    ).fetchall()

    total_updates = 0
    for film in films:
        film_id, title, filepath, fps = (
            film["film_id"], film["title"], film["filepath"], film["fps"] or 12.0
        )
        stats = conn.execute(
            "SELECT COUNT(*) n, MAX(frame_idx) mx FROM poses WHERE film_id=?", (film_id,)
        ).fetchone()
        if not stats["n"]:
            continue

        max_idx = stats["mx"]
        if max_idx < 9999:
            print(f"[skip] {title[:52]:<52} {stats['n']:>6} poses  (max idx {max_idx}, unaffected)")
            continue

        if not Path(filepath).exists():
            print(f"[MISS] {title[:52]:<52} source file not found, cannot repair")
            continue

        duration = probe_duration(filepath)
        if not duration:
            print(f"[MISS] {title[:52]:<52} ffprobe failed, cannot repair")
            continue

        # ffmpeg's fps filter emits about duration*fps frames; the exact count
        # decides the permutation, so search a small window for the best fit.
        estimate = int(round(duration * fps))
        best_n, best_score = None, -1.0
        for candidate_n in range(max(max_idx + 1, estimate - 6), estimate + 7):
            mapping = sorted_position_to_frame_number(candidate_n)
            if len(mapping) <= max_idx:
                continue
            score = score_mapping(conn, film_id, mapping, fps, samples)
            if score > best_score:
                best_n, best_score = candidate_n, score

        if best_n is None:
            print(f"[MISS] {title[:52]:<52} no viable frame count found")
            continue

        mapping = sorted_position_to_frame_number(best_n)
        rows = conn.execute(
            "SELECT pose_id, frame_idx, timestamp FROM poses WHERE film_id=?", (film_id,)
        ).fetchall()

        updates: List[Tuple[float, int]] = []
        for row in rows:
            idx = row["frame_idx"]
            if 0 <= idx < len(mapping):
                updates.append(((mapping[idx] - 1) / fps, row["pose_id"]))

        print(
            f"[FIX ] {title[:52]:<52} {len(updates):>6} poses  "
            f"n={best_n} (est {estimate}) monotonic={best_score:.3f}"
        )
        if apply:
            conn.executemany(
                "UPDATE poses SET timestamp=? WHERE pose_id=?", updates
            )
            total_updates += len(updates)

    if apply:
        conn.commit()
        print(f"\nApplied {total_updates} timestamp corrections to {db_path}")
    else:
        print("\nDry run - pass --apply to write these corrections")
    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="pose_library.db", help="Path to pose database")
    parser.add_argument("--apply", action="store_true", help="Write corrections (default: dry run)")
    args = parser.parse_args()
    repair(args.db, args.apply)


if __name__ == "__main__":
    main()
