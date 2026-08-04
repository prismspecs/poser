#!/usr/bin/env python3
"""
Database Cleaning Script
========================
Validates every pose in the database against YOLO segmentation model.
Removes poses where the source frame does NOT contain a Class 0 (person) detection
with sufficient confidence.

This fixes the core problem: YOLO pose estimation happily fits skeletons to 
non-human objects (AI-generated imagery, abstract shapes, etc.), and those 
garbage poses pollute the matching results.
"""

import sqlite3
import sys
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

from utils.image_utils import extract_frame_at_timestamp

# Minimum confidence for person (Class 0) detection
PERSON_CONFIDENCE_THRESHOLD = 0.40

# Minimum bounding box area ratio (person bbox vs image area)
MIN_PERSON_AREA_RATIO = 0.01


def verify_person_in_frame(seg_model, frame, bbox=None, conf_threshold=PERSON_CONFIDENCE_THRESHOLD):
    """
    Verify that a real person exists in the frame at/near the stored bounding box.
    
    Returns: (is_person: bool, best_person_conf: float, best_iou: float)
    """
    results = seg_model(frame, verbose=False)
    
    best_conf = 0.0
    best_iou = 0.0
    
    for r in results:
        if r.boxes is None or len(r.boxes.cls) == 0:
            continue
        for i, cls_id in enumerate(r.boxes.cls):
            if int(cls_id) != 0:  # Not a person
                continue
            conf = float(r.boxes.conf[i])
            if conf < conf_threshold:
                continue
            
            best_conf = max(best_conf, conf)
            
            # If we have stored bbox, compute IoU to verify it's the SAME person
            if bbox is not None:
                det_box = r.boxes.xyxy[i].cpu().numpy()
                iou = compute_iou(bbox, det_box)
                best_iou = max(best_iou, iou)
            else:
                best_iou = 1.0  # No bbox to compare, just trust the detection
    
    is_person = best_conf >= conf_threshold
    return is_person, best_conf, best_iou


def compute_iou(box_a, box_b):
    """Compute Intersection over Union between two [x1,y1,x2,y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def clean_database(db_path="pose_library.db", dry_run=False, batch_size=50):
    """
    Scan all poses in the database, verify each one contains a real person,
    and delete the ones that don't.
    """
    conn = sqlite3.connect(db_path)
    
    # Load segmentation model
    print("Loading YOLO segmentation model...")
    seg_model = YOLO("yolo11n-seg.pt")
    
    # Get all films
    films = conn.execute("SELECT film_id, title, filepath FROM films").fetchall()
    print(f"Found {len(films)} films in database")
    
    total_deleted = 0
    total_checked = 0
    
    for film_id, title, filepath in films:
        if not Path(filepath).exists():
            print(f"\n⚠️  Film file missing: {title} ({filepath})")
            pose_count = conn.execute("SELECT COUNT(*) FROM poses WHERE film_id = ?", (film_id,)).fetchone()[0]
            if pose_count > 0:
                print(f"   Deleting {pose_count} poses (source file unavailable)")
                if not dry_run:
                    conn.execute("DELETE FROM poses WHERE film_id = ?", (film_id,))
                    conn.commit()
                total_deleted += pose_count
            continue
        
        # Get all poses for this film
        poses = conn.execute(
            "SELECT pose_id, timestamp, bbox_x1, bbox_y1, bbox_x2, bbox_y2, confidence FROM poses WHERE film_id = ?",
            (film_id,)
        ).fetchall()

        if not poses:
            continue

        print(f"\nVerifying {len(poses)} poses from: {title}")

        # Group by timestamp to avoid re-reading the same frame. Grouping by
        # frame_idx and seeking to it would read the wrong moment entirely:
        # frame_idx counts resampled ingest frames, not native video frames.
        frame_groups = {}
        for pose_id, timestamp, bx1, by1, bx2, by2, conf in poses:
            if timestamp not in frame_groups:
                frame_groups[timestamp] = []
            frame_groups[timestamp].append({
                "pose_id": pose_id,
                "bbox": [bx1, by1, bx2, by2] if all(v is not None for v in [bx1, by1, bx2, by2]) else None,
                "confidence": conf
            })
        
        to_delete = []
        verified = 0
        
        timestamps = sorted(frame_groups.keys())
        for timestamp in tqdm(timestamps, desc=f"  {title[:40]}"):
            frame = extract_frame_at_timestamp(filepath, timestamp)
            if frame is None:
                # Can't read frame -> delete these poses
                for p in frame_groups[timestamp]:
                    to_delete.append(p["pose_id"])
                continue
            
            # Run YOLO segmentation once per frame
            seg_results = seg_model(frame, verbose=False)
            
            # Collect all person detections in this frame
            person_detections = []
            for r in seg_results:
                if r.boxes is None or len(r.boxes.cls) == 0:
                    continue
                for i, cls_id in enumerate(r.boxes.cls):
                    if int(cls_id) == 0:
                        conf = float(r.boxes.conf[i])
                        det_box = r.boxes.xyxy[i].cpu().numpy()
                        person_detections.append((conf, det_box))
            
            # Check each pose against person detections
            for p in frame_groups[timestamp]:
                total_checked += 1
                is_valid = False
                
                if not person_detections:
                    # No person at all in this frame
                    is_valid = False
                elif p["bbox"] is not None:
                    # Check if any person detection overlaps with stored bbox
                    for det_conf, det_box in person_detections:
                        if det_conf < PERSON_CONFIDENCE_THRESHOLD:
                            continue
                        iou = compute_iou(p["bbox"], det_box)
                        if iou > 0.2:  # Some overlap means it's the same person
                            is_valid = True
                            break
                    if not is_valid:
                        # Fallback: any high-confidence person detection
                        for det_conf, _ in person_detections:
                            if det_conf >= 0.6:
                                is_valid = True
                                break
                else:
                    # No bbox stored, just check if there's any person
                    for det_conf, _ in person_detections:
                        if det_conf >= PERSON_CONFIDENCE_THRESHOLD:
                            is_valid = True
                            break
                
                if is_valid:
                    verified += 1
                else:
                    to_delete.append(p["pose_id"])
        
        if to_delete:
            print(f"  ❌ Removing {len(to_delete)} non-human poses (keeping {verified})")
            if not dry_run:
                # Delete in batches
                for i in range(0, len(to_delete), batch_size):
                    batch = to_delete[i:i+batch_size]
                    placeholders = ",".join(["?"] * len(batch))
                    conn.execute(f"DELETE FROM poses WHERE pose_id IN ({placeholders})", batch)
                conn.commit()
            total_deleted += len(to_delete)
        else:
            print(f"  ✅ All {verified} poses verified as human")
    
    # Summary
    total_remaining = conn.execute("SELECT COUNT(*) FROM poses").fetchone()[0]
    print(f"\n{'='*50}")
    print(f"CLEANING SUMMARY {'(DRY RUN)' if dry_run else ''}")
    print(f"{'='*50}")
    print(f"Total poses checked:  {total_checked}")
    print(f"Total poses deleted:  {total_deleted}")
    print(f"Poses remaining:      {total_remaining}")
    print(f"{'='*50}")
    
    conn.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Clean non-human poses from database")
    parser.add_argument("--db", default="pose_library.db", help="Path to pose database")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without deleting")
    args = parser.parse_args()
    
    clean_database(db_path=args.db, dry_run=args.dry_run)
