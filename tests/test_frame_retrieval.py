"""
Regression tests for source-frame retrieval.

Both bugs covered here silently produced plausible-looking video: the pipeline
still composited *a* person, just the wrong one from the wrong moment, so only
an explicit test catches a reintroduction.
"""

import subprocess

import cv2
import numpy as np
import pytest

from utils.image_utils import extract_frame_at_timestamp, natural_frame_key


def test_natural_frame_key_orders_past_four_digits():
    """ffmpeg widens frame_%04d past 9999; plain sorting scrambles that."""
    names = [f"frame_{i:04d}.jpg" for i in range(1, 12005)]

    lexicographic = sorted(names)
    natural = sorted(names, key=natural_frame_key)

    # The bug: frame_10000 lands right after frame_1000, ~11000 slots too early.
    assert lexicographic.index("frame_10000.jpg") == 1000
    assert lexicographic[-1] == "frame_9999.jpg"

    # Natural ordering keeps position == frame number - 1 throughout.
    assert natural == [f"frame_{i:04d}.jpg" for i in range(1, 12005)]
    assert natural.index("frame_10000.jpg") == 9999


def test_natural_frame_key_handles_paths_and_missing_digits():
    """The key must accept full paths and not crash on unnumbered names."""
    assert natural_frame_key("a/b/frame_0042.jpg") < natural_frame_key("a/b/frame_0043.jpg")
    assert natural_frame_key("frame_0007.png") == natural_frame_key("other/frame_0007.jpg")
    # No digits sorts first rather than raising.
    assert natural_frame_key("frame_.jpg")[0] == -1


@pytest.fixture
def counter_video(tmp_path):
    """
    Build a 6-second, 30fps video whose green channel encodes its frame number.

    This gives every frame a machine-readable identity, so a seek can be checked
    against the exact frame it was supposed to land on.
    """
    path = tmp_path / "counter.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 64)
    )
    if not writer.isOpened():
        pytest.skip("No usable mp4 encoder available")

    total = 180
    for i in range(total):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[:, :] = (0, i, 0)  # green channel == frame index
        writer.write(frame)
    writer.release()

    if not path.exists() or path.stat().st_size == 0:
        pytest.skip("Video encoding produced no output")
    return path, total


def test_extract_frame_at_timestamp_lands_on_the_right_frame(counter_video):
    """A timestamp must resolve to the frame actually shown at that time."""
    path, _ = counter_video

    for seconds, expected_index in [(0.0, 0), (1.0, 30), (2.0, 60), (4.0, 120)]:
        frame = extract_frame_at_timestamp(path, seconds)
        assert frame is not None, f"no frame returned at {seconds}s"
        decoded = int(round(float(np.median(frame[:, :, 1]))))
        assert abs(decoded - expected_index) <= 1, (
            f"seek to {seconds}s returned frame {decoded}, expected {expected_index}"
        )


def test_timestamp_seek_differs_from_resampled_index(counter_video):
    """
    Guard the actual regression: ingest indexes resampled frames, so treating a
    frame_idx as a native frame number reads the wrong moment entirely.
    """
    path, _ = counter_video
    ingest_fps = 12.0
    frame_idx = 48  # 48th frame of a 12fps resampling == 4.0s == native frame 120

    timestamp = frame_idx / ingest_fps
    correct = extract_frame_at_timestamp(path, timestamp)
    assert correct is not None

    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)  # the old, incorrect behaviour
    ok, naive = cap.read()
    cap.release()
    assert ok

    correct_index = float(np.median(correct[:, :, 1]))
    naive_index = float(np.median(naive[:, :, 1]))

    assert abs(correct_index - 120) <= 1
    assert abs(naive_index - 48) <= 1
    assert abs(correct_index - naive_index) > 60


def test_extract_frame_at_timestamp_returns_none_for_bad_input(tmp_path):
    """Unreadable sources must return None rather than raising."""
    assert extract_frame_at_timestamp(tmp_path / "nope.mp4", 1.0) is None
