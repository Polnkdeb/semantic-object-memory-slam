from pathlib import Path
import json
import sys

import cv2
import numpy as np
from tqdm import tqdm


def get_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(3000)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors


def match_features(des1, des2):
    if des1 is None or des2 is None:
        return []

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches[:800]


def add_pose(poses, frame_name, t_total):
    poses.append({
        "frame": frame_name,
        "x": float(t_total[0, 0]),
        "y": float(t_total[1, 0]),
        "z": float(t_total[2, 0]),
    })


def main():
    if len(sys.argv) != 3:
        print("Usage: python src/visual_odometry_scene.py <frames_dir> <poses_json>")
        return

    frames_dir = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(frames_dir.glob("*.jpg"))

    if len(frame_paths) < 2:
        raise RuntimeError("Need at least 2 frames for visual odometry")

    first_img = cv2.imread(str(frame_paths[0]))
    h, w = first_img.shape[:2]

    focal = 0.8 * w
    cx, cy = w / 2, h / 2

    K = np.array([
        [focal, 0, cx],
        [0, focal, cy],
        [0, 0, 1],
    ])

    R_total = np.eye(3)
    t_total = np.zeros((3, 1))

    poses = [{
        "frame": frame_paths[0].name,
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
    }]

    prev_img = first_img
    prev_kp, prev_des = get_features(prev_img)

    for frame_path in tqdm(frame_paths[1:], desc="Visual odometry"):
        curr_img = cv2.imread(str(frame_path))
        curr_kp, curr_des = get_features(curr_img)

        matches = match_features(prev_des, curr_des)

        if len(matches) < 20:
            add_pose(poses, frame_path.name, t_total)
            prev_kp, prev_des = curr_kp, curr_des
            continue

        pts1 = np.float32([prev_kp[m.queryIdx].pt for m in matches])
        pts2 = np.float32([curr_kp[m.trainIdx].pt for m in matches])

        E, mask = cv2.findEssentialMat(
            pts1,
            pts2,
            K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
        )

        if E is None:
            add_pose(poses, frame_path.name, t_total)
            prev_kp, prev_des = curr_kp, curr_des
            continue

        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)

        t_total = t_total + R_total @ t
        R_total = R @ R_total

        add_pose(poses, frame_path.name, t_total)

        prev_kp, prev_des = curr_kp, curr_des

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(poses, f, indent=2)

    print(f"Saved poses to {out_path}")
    print(f"Total poses: {len(poses)}")


if __name__ == "__main__":
    main()