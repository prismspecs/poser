"""
Tests for person-cutout compositing.

These cover the guards that decide whether a match is renderable at all. Each
one exists because its failure mode reached the finished video: a rectangle of
raw film footage, a body blown up past recognition, or a whole figure pasted
onto a close-up of someone's feet.
"""

import numpy as np
import pytest

from pose_visualizer import PoseVisualizer
from utils.pose_utils import PoseData


def make_pose(center=(500.0, 900.0), torso_half=60.0, torso_points=4, bbox=None):
    """
    Build a PoseData with a controllable torso quad.

    Args:
        center: (x, y) midpoint of the shoulder/hip block.
        torso_half: Half-width/height of the torso quad, setting pose scale.
        torso_points: How many of the 4 torso keypoints are present.
        bbox: Optional explicit bounding box.

    Returns:
        PoseData with 17 keypoint slots, non-torso entries left as None.
    """
    cx, cy = center
    keypoints = [None] * 17
    # indices 5,6 = shoulders; 11,12 = hips
    layout = {
        5: (cx - torso_half, cy - torso_half),
        6: (cx + torso_half, cy - torso_half),
        11: (cx - torso_half, cy + torso_half),
        12: (cx + torso_half, cy + torso_half),
    }
    for i, (idx, (x, y)) in enumerate(layout.items()):
        if i < torso_points:
            keypoints[idx] = (x, y, 0.95)

    if bbox is None:
        bbox = (cx - torso_half * 1.5, cy - torso_half * 3, cx + torso_half * 1.5, cy + torso_half * 3)

    return PoseData(
        keypoints=keypoints,
        bounding_box=bbox,
        confidence_score=0.9,
        image_path="synthetic",
        pose_id="synthetic",
    )


class StubSegmentation:
    """Stands in for the YOLO segmentation model, with controllable output."""

    def __init__(self, mask=None, boxes=None):
        self._mask = mask
        self._boxes = boxes

    def __call__(self, image, verbose=False):
        if self._mask is None:
            return [_StubResult(None, None)]
        return [_StubResult(self._mask, self._boxes)]


class _StubBox:
    def __init__(self, xyxy, cls=0):
        self.xyxy = [_ToNumpy(np.array(xyxy, dtype=np.float32))]
        self.cls = [cls]


class _ToNumpy:
    def __init__(self, arr):
        self._arr = arr

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _StubMasks:
    def __init__(self, mask):
        self.data = [_ToNumpy(mask.astype(np.float32))]


class _StubResult:
    def __init__(self, mask, boxes):
        self.masks = _StubMasks(mask) if mask is not None else None
        self.boxes = boxes


@pytest.fixture
def visualizer():
    return PoseVisualizer()


@pytest.fixture
def frames():
    """A target frame and a source frame, both plain colours."""
    target = np.full((1920, 1080, 3), 40, dtype=np.uint8)
    source = np.full((1080, 1920, 3), 200, dtype=np.uint8)
    return target, source


def test_returns_none_when_segmentation_fails(visualizer, frames):
    """
    A failed segmentation must abort, not paste the bounding box.

    The box fallback rendered an opaque rectangle of unrelated footage over the
    frame, which is far more conspicuous than simply skipping the match.
    """
    target, source = frames
    result = visualizer.create_person_cutout_composite(
        target_image=target,
        target_pose=make_pose(),
        source_image=source,
        source_bbox=(800.0, 300.0, 1000.0, 800.0),
        segmentation_model=StubSegmentation(mask=None),
        source_pose=make_pose(center=(900.0, 550.0)),
    )
    assert result is None


def test_returns_none_without_target_torso(visualizer, frames):
    """Close-ups lacking a torso must be skipped, not given a whole body."""
    target, source = frames
    result = visualizer.create_person_cutout_composite(
        target_image=target,
        target_pose=make_pose(torso_points=1),  # only one torso keypoint survived
        source_image=source,
        source_bbox=(800.0, 300.0, 1000.0, 800.0),
        segmentation_model=StubSegmentation(mask=None),
        source_pose=make_pose(center=(900.0, 550.0)),
    )
    assert result is None


def test_returns_none_on_extreme_magnification(visualizer, frames):
    """A tiny distant source scaled onto a huge target smears into a blob."""
    target, source = frames
    mask = np.zeros((1080, 1920), dtype=np.uint8)
    mask[300:800, 800:1000] = 1

    result = visualizer.create_person_cutout_composite(
        target_image=target,
        target_pose=make_pose(torso_half=300.0),   # very large target body
        source_image=source,
        source_bbox=(800.0, 300.0, 1000.0, 800.0),
        segmentation_model=StubSegmentation(mask=mask),
        source_pose=make_pose(center=(900.0, 550.0), torso_half=4.0),  # tiny source
        max_scale=6.0,
    )
    assert result is None


def test_composites_and_preserves_target_resolution(visualizer, frames):
    """
    A good match composites source pixels at the target's own resolution.

    Output was previously forced into a 1920x1080 letterbox, which pillarboxed
    vertical footage with black bars.
    """
    target, source = frames
    mask = np.zeros((1080, 1920), dtype=np.uint8)
    mask[300:800, 800:1000] = 1
    boxes = [_StubBox([800, 300, 1000, 800])]

    result = visualizer.create_person_cutout_composite(
        target_image=target,
        target_pose=make_pose(center=(540.0, 900.0), torso_half=80.0),
        source_image=source,
        source_bbox=(800.0, 300.0, 1000.0, 800.0),
        segmentation_model=StubSegmentation(mask=mask, boxes=boxes),
        source_pose=make_pose(center=(900.0, 550.0), torso_half=60.0),
    )

    assert result is not None
    assert result.shape == target.shape, "composite must keep the target's resolution"
    # Source pixels (200) are brighter than the target background (40).
    assert result.max() > 150, "source person should be visible in the composite"
    assert (result == 40).any(), "target background should remain outside the cutout"


def test_returns_none_when_cutout_swallows_the_frame(visualizer, frames):
    """
    A source close-up scaled up covers most of the frame as an abstract smear.

    Real bodies measured under 0.32 of the frame across a full reconstruction;
    these artefacts sat above 0.49, so the cap must reject them.
    """
    target, source = frames
    mask = np.zeros((1080, 1920), dtype=np.uint8)
    mask[:, :] = 1  # segmentation covers essentially the whole source frame
    boxes = [_StubBox([0, 0, 1920, 1080])]

    result = visualizer.create_person_cutout_composite(
        target_image=target,
        target_pose=make_pose(center=(540.0, 900.0), torso_half=80.0),
        source_image=source,
        source_bbox=(0.0, 0.0, 1920.0, 1080.0),
        segmentation_model=StubSegmentation(mask=mask, boxes=boxes),
        source_pose=make_pose(center=(960.0, 540.0), torso_half=80.0),
    )
    assert result is None


def test_pose_anchor_uses_torso_not_bbox(visualizer):
    """
    Scale must come from the torso, so flailing limbs do not resize the cutout.

    Two poses share a torso but differ wildly in bounding box; their anchors
    must be identical.
    """
    compact = make_pose(center=(500.0, 900.0), torso_half=60.0, bbox=(440.0, 720.0, 560.0, 1080.0))
    sprawling = make_pose(center=(500.0, 900.0), torso_half=60.0, bbox=(0.0, 0.0, 1080.0, 1920.0))

    assert visualizer._pose_anchor(compact) == visualizer._pose_anchor(sprawling)


def test_pose_anchor_rejects_sparse_torso(visualizer):
    """Fewer than three torso keypoints is not enough to place a body."""
    assert visualizer._pose_anchor(make_pose(torso_points=2)) is None
    assert visualizer._pose_anchor(make_pose(torso_points=3)) is not None
    assert visualizer._pose_anchor(None) is None
