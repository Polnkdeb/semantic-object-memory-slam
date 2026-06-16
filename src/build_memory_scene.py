from pathlib import Path
import json
import sys


def main():
    if len(sys.argv) != 4:
        print(
            "Usage: python src/build_memory_scene.py "
            "<detections_json> <poses_json> <memory_json>"
        )
        return

    detections_path = Path(sys.argv[1])
    poses_path = Path(sys.argv[2])
    out_path = Path(sys.argv[3])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(detections_path, "r", encoding="utf-8") as f:
        detections = json.load(f)

    with open(poses_path, "r", encoding="utf-8") as f:
        poses = json.load(f)

    pose_by_frame = {pose["frame"]: pose for pose in poses}

    memory = []

    for frame_name, objects in detections.items():
        pose = pose_by_frame.get(frame_name)

        if pose is None:
            continue

        for obj in objects:
            memory.append({
                "label": obj["class"],
                "confidence": obj["confidence"],
                "frame": frame_name,
                "bbox": obj["bbox"],
                "camera_position": {
                    "x": pose["x"],
                    "y": pose["y"],
                    "z": pose["z"],
                },
            })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2)

    print(f"Saved object memory to {out_path}")
    print(f"Remembered observations: {len(memory)}")


if __name__ == "__main__":
    main()