#!/usr/bin/env python3
"""
Pipeline Component Tests
========================
Validates each component of the pose matching pipeline independently:
1. Person Detection - can we tell humans from non-humans?
2. Pose Extraction - are we getting real skeletons from real people?
3. Matching Quality - does pose similarity actually match similar poses?
4. Compositing - does the cutout/overlay look correct?

Run: python test_pipeline.py
"""

import cv2
import numpy as np
import sys
from pathlib import Path
from ultralytics import YOLO

PERSON_CONF_THRESHOLD = 0.40


def extract_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def test_person_detection():
    """Test 1: Verify YOLO segmentation can distinguish people from non-people."""
    print("\n" + "="*60)
    print("TEST 1: PERSON DETECTION ACCURACY")
    print("="*60)
    
    seg_model = YOLO("yolo11n-seg.pt")
    
    # Known sources and sample frames to test
    test_cases = []
    
    # Silo - TV show, should have lots of real people
    silo_path = "/Users/grayson/Downloads/Silo.S03E03.1080p.WEB.h264-ETHEL[EZTVx.to].mkv"
    if Path(silo_path).exists():
        for fidx in [1000, 5000, 10000, 15000, 20000]:
            test_cases.append(("Silo", silo_path, fidx, True))
    
    # MidJourney1 - AI art, mix of people and non-people
    mj_path = "/Users/grayson/Downloads/MidJourney1.mp4"
    if Path(mj_path).exists():
        # These specific frames were flagged as NOT HUMAN in our trace
        for fidx in [1331, 1334, 1335, 1336, 1337]:
            test_cases.append(("MidJourney1-suspect", mj_path, fidx, False))
        # Some MJ frames DO have people
        for fidx in [1069, 1111]:
            test_cases.append(("MidJourney1-person?", mj_path, fidx, None))
    
    # Phone video - mostly not people
    phone_path = "/Users/grayson/Downloads/the phone video to intercut.mp4"
    if Path(phone_path).exists():
        for fidx in [3140, 3153, 3157]:
            test_cases.append(("PhoneVid-suspect", phone_path, fidx, False))
    
    # banker-masked - should be people
    banker_path = "/Users/grayson/Downloads/banker-masked(1).mp4"
    if Path(banker_path).exists():
        for fidx in [500, 1000, 1500]:
            test_cases.append(("Banker", banker_path, fidx, True))
    
    if not test_cases:
        print("  ⚠️  No test videos found, skipping")
        return False
    
    passed = 0
    failed = 0
    unclear = 0
    
    for label, vpath, fidx, expected_person in test_cases:
        frame = extract_frame(vpath, fidx)
        if frame is None:
            print(f"  SKIP {label} frame={fidx} (couldn't read)")
            continue
        
        results = seg_model(frame, verbose=False)
        best_person_conf = 0.0
        detected_classes = []
        for r in results:
            if r.boxes is not None:
                for i, cls_id in enumerate(r.boxes.cls):
                    cid = int(cls_id)
                    conf = float(r.boxes.conf[i])
                    detected_classes.append((cid, conf))
                    if cid == 0:
                        best_person_conf = max(best_person_conf, conf)
        
        is_person = best_person_conf >= PERSON_CONF_THRESHOLD
        
        if expected_person is None:
            status = "ℹ️ "
            unclear += 1
        elif is_person == expected_person:
            status = "✅"
            passed += 1
        else:
            status = "❌"
            failed += 1
        
        classes_str = ", ".join(f"cls{c}({cf:.2f})" for c, cf in detected_classes[:5])
        print(f"  {status} {label:25s} frame={fidx:6d} | person_conf={best_person_conf:.3f} | detected=[{classes_str}]")
    
    print(f"\n  Results: {passed} passed, {failed} failed, {unclear} unclear")
    return failed == 0


def test_pose_extraction():
    """Test 2: Verify pose estimation extracts valid skeletons from real people."""
    print("\n" + "="*60)
    print("TEST 2: POSE EXTRACTION QUALITY")
    print("="*60)
    
    pose_model = YOLO("yolo11n-pose.pt")
    seg_model = YOLO("yolo11n-seg.pt")
    
    # Use the dance target as ground truth - we know it has a person dancing
    target_path = "data/test_dance_target.mp4"
    if not Path(target_path).exists():
        # Fallback to input.mkv
        target_path = "test-vids/input.mkv"
    
    if not Path(target_path).exists():
        print("  ⚠️  No test video found, skipping")
        return False
    
    passed = 0
    failed = 0
    
    for fidx in range(0, 150, 10):
        frame = extract_frame(target_path, fidx)
        if frame is None:
            continue
        
        # Extract pose
        pose_results = pose_model(frame, verbose=False)
        has_pose = False
        has_torso = False
        keypoint_count = 0
        pose_conf = 0.0
        
        for r in pose_results:
            if r.keypoints is not None and len(r.keypoints.data) > 0:
                for person_kps in r.keypoints.data:
                    visible_kps = (person_kps[:, 2] > 0.3).sum().item()
                    if visible_kps > keypoint_count:
                        keypoint_count = visible_kps
                        has_pose = True
                        # Check torso keypoints (5=lshoulder, 6=rshoulder, 11=lhip, 12=rhip)
                        torso_visible = all(person_kps[i, 2] > 0.3 for i in [5, 6, 11, 12])
                        has_torso = torso_visible
                if r.boxes is not None and len(r.boxes.conf) > 0:
                    pose_conf = float(r.boxes.conf.max())
        
        # Also verify with segmentation
        seg_results = seg_model(frame, verbose=False)
        seg_person = False
        seg_conf = 0.0
        for r in seg_results:
            if r.boxes is not None:
                for i, cls in enumerate(r.boxes.cls):
                    if int(cls) == 0:
                        c = float(r.boxes.conf[i])
                        if c >= PERSON_CONF_THRESHOLD:
                            seg_person = True
                            seg_conf = max(seg_conf, c)
        
        # A good frame should have: pose + torso + segmentation person
        all_good = has_pose and has_torso and seg_person
        status = "✅" if all_good else "❌"
        if all_good:
            passed += 1
        else:
            failed += 1
        
        print(f"  {status} frame={fidx:4d} | pose={'Y' if has_pose else 'N'} torso={'Y' if has_torso else 'N'} "
              f"kps={keypoint_count:2d} pose_conf={pose_conf:.2f} | seg_person={'Y' if seg_person else 'N'} seg_conf={seg_conf:.2f}")
    
    print(f"\n  Results: {passed} passed, {failed} failed")
    return failed <= 2  # Allow a couple failures (some frames may have no person)


def test_pose_matching():
    """Test 3: Verify pose similarity math works correctly."""
    print("\n" + "="*60)
    print("TEST 3: POSE MATCHING SIMILARITY")
    print("="*60)
    
    from pose_db import PoseDatabase
    from pose_estimator import PoseEstimator
    
    estimator = PoseEstimator(model_size='n')
    
    target_path = "data/test_dance_target.mp4"
    if not Path(target_path).exists():
        target_path = "test-vids/input.mkv"
    if not Path(target_path).exists():
        print("  ⚠️  No test video found, skipping")
        return False
    
    # Extract poses from two nearby frames (should be similar) and two distant frames (should differ)
    frames = {}
    for fidx in [0, 1, 50, 100]:
        frame = extract_frame(target_path, fidx)
        if frame is None:
            continue
        poses = estimator.extract_poses(frame, f"data/tmp_test_{fidx}.png")
        if poses:
            best = max(poses, key=lambda p: p.confidence_score)
            vec = PoseDatabase.normalize_keypoints_to_vector(best.keypoints)
            if vec is not None:
                frames[fidx] = vec
    
    if len(frames) < 3:
        print("  ⚠️  Not enough frames extracted, skipping")
        return False
    
    # Compute MSE similarities
    def mse_similarity(v1, v2):
        a = np.array(v1, dtype=np.float32)
        b = np.array(v2, dtype=np.float32)
        mse = np.mean((a - b) ** 2)
        return max(0, 1.0 - mse)
    
    passed = 0
    failed = 0
    
    if 0 in frames and 1 in frames:
        sim_nearby = mse_similarity(frames[0], frames[1])
        ok = sim_nearby > 0.8
        status = "✅" if ok else "❌"
        if ok: passed += 1
        else: failed += 1
        print(f"  {status} Nearby frames (0 vs 1):   similarity = {sim_nearby:.4f} (expect > 0.8)")
    
    if 0 in frames and 50 in frames:
        sim_mid = mse_similarity(frames[0], frames[50])
        print(f"  ℹ️  Mid-distance (0 vs 50):  similarity = {sim_mid:.4f}")
    
    if 0 in frames and 100 in frames:
        sim_far = mse_similarity(frames[0], frames[100])
        print(f"  ℹ️  Far frames (0 vs 100):   similarity = {sim_far:.4f}")
    
    # Self-similarity should be 1.0
    if 0 in frames:
        sim_self = mse_similarity(frames[0], frames[0])
        ok = abs(sim_self - 1.0) < 0.001
        status = "✅" if ok else "❌"
        if ok: passed += 1
        else: failed += 1
        print(f"  {status} Self-similarity (0 vs 0): similarity = {sim_self:.4f} (expect = 1.0)")
    
    print(f"\n  Results: {passed} passed, {failed} failed")
    return failed == 0


def test_compositing():
    """Test 4: Verify person segmentation and compositing produces valid output."""
    print("\n" + "="*60)
    print("TEST 4: COMPOSITING QUALITY")
    print("="*60)
    
    from pose_estimator import PoseEstimator
    from pose_visualizer import PoseVisualizer
    
    estimator = PoseEstimator(model_size='n')
    visualizer = PoseVisualizer()
    
    # Use a known-good source: banker-masked has 90% person rate
    source_path = "/Users/grayson/Downloads/banker-masked(1).mp4"
    target_path = "data/test_dance_target.mp4"
    if not Path(target_path).exists():
        target_path = "test-vids/input.mkv"
    
    if not Path(source_path).exists() or not Path(target_path).exists():
        print("  ⚠️  Test videos not found, skipping")
        return False
    
    target_frame = extract_frame(target_path, 30)
    source_frame = extract_frame(source_path, 500)
    
    if target_frame is None or source_frame is None:
        print("  ⚠️  Couldn't read test frames, skipping")
        return False
    
    # Get target pose
    target_poses = estimator.extract_poses(target_frame, "data/tmp_comp_target.png")
    if not target_poses:
        print("  ❌ No pose detected in target frame")
        return False
    target_pose = max(target_poses, key=lambda p: p.confidence_score)
    
    # Get source bbox via segmentation
    seg_results = estimator.segmentation_model(source_frame, verbose=False)
    source_bbox = None
    for r in seg_results:
        if r.boxes is not None:
            for i, cls_id in enumerate(r.boxes.cls):
                if int(cls_id) == 0 and float(r.boxes.conf[i]) >= 0.40:
                    box = r.boxes.xyxy[i].cpu().numpy()
                    source_bbox = [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
                    break
    
    if source_bbox is None:
        print("  ❌ No person detected in source frame for bbox")
        return False
    
    print(f"  ✅ Target pose extracted ({len(target_pose.keypoints)} keypoints)")
    print(f"  ✅ Source person bbox: [{source_bbox[0]:.0f}, {source_bbox[1]:.0f}, {source_bbox[2]:.0f}, {source_bbox[3]:.0f}]")
    
    # Test composite
    try:
        composite = visualizer.create_person_cutout_composite(
            target_image=target_frame,
            target_pose=target_pose,
            source_image=source_frame,
            source_bbox=source_bbox,
            segmentation_model=estimator.segmentation_model
        )
        
        if composite is not None and composite.shape[0] > 0 and composite.shape[1] > 0:
            out_path = "results/test_composite.png"
            Path("results").mkdir(exist_ok=True)
            cv2.imwrite(out_path, composite)
            print(f"  ✅ Composite created: {composite.shape} -> {out_path}")
            
            # Basic sanity: composite shouldn't be all black or all one color
            mean_val = composite.mean()
            std_val = composite.std()
            if std_val < 5:
                print(f"  ❌ Composite looks blank (std={std_val:.1f})")
                return False
            print(f"  ✅ Composite looks reasonable (mean={mean_val:.1f}, std={std_val:.1f})")
        else:
            print("  ❌ Composite is None or empty")
            return False
    except Exception as e:
        print(f"  ❌ Composite failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test diagnostic view too
    try:
        source_poses = estimator.extract_poses(source_frame, "data/tmp_comp_source.png")
        s_pose = max(source_poses, key=lambda p: p.confidence_score) if source_poses else None
        diag = visualizer.create_side_by_side_diagnostic(
            target_image=target_frame,
            target_pose=target_pose,
            source_image=source_frame,
            source_pose=s_pose,
            film_title="banker-masked(1)",
            frame_idx=500,
            similarity_score=0.85,
            frame_num=1
        )
        if diag is not None:
            cv2.imwrite("results/test_diagnostic.png", diag)
            print(f"  ✅ Diagnostic view created: {diag.shape}")
        else:
            print("  ❌ Diagnostic view is None")
            return False
    except Exception as e:
        print(f"  ❌ Diagnostic view failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    print("🔬 POSER PIPELINE COMPONENT TESTS")
    print("=" * 60)
    
    results = {}
    
    results["Person Detection"] = test_person_detection()
    results["Pose Extraction"] = test_pose_extraction()
    results["Pose Matching"] = test_pose_matching()
    results["Compositing"] = test_compositing()
    
    print("\n" + "=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} - {name}")
    
    all_passed = all(results.values())
    print(f"\n{'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    sys.exit(0 if all_passed else 1)
