import json
import sys
from pathlib import Path
from math import sqrt


DISTANCE_THRESHOLD = 2.0


def distance_3d(a, b):
    return sqrt(
        (a["x"] - b["x"]) ** 2 +
        (a["y"] - b["y"]) ** 2 +
        (a["z"] - b["z"]) ** 2
    )


def main():
    if len(sys.argv) != 3:
        print("Usage: python src/object_identity_scene.py <memory_json> <objects_json>")
        return

    memory_path = Path(sys.argv[1])
    objects_path = Path(sys.argv[2])

    with open(memory_path, "r", encoding="utf-8") as f:
        observations = json.load(f)

    objects = []
    next_object_id = 0

    for obs in observations:
        label = obs["label"]
        pos = obs["camera_position"]

        best_obj = None
        best_dist = float("inf")

        for obj in objects:
            if obj["label"] != label:
                continue

            d = distance_3d(pos, obj["mean_position"])

            if d < best_dist:
                best_dist = d
                best_obj = obj

        if best_obj is not None and best_dist < DISTANCE_THRESHOLD:
            best_obj["observations"].append(obs)
            n = len(best_obj["observations"])

            old = best_obj["mean_position"]
            best_obj["mean_position"] = {
                "x": old["x"] + (pos["x"] - old["x"]) / n,
                "y": old["y"] + (pos["y"] - old["y"]) / n,
                "z": old["z"] + (pos["z"] - old["z"]) / n,
            }

            best_obj["max_confidence"] = max(
                best_obj["max_confidence"],
                obs["confidence"]
            )

            best_obj["frames"].append(obs["frame"])

        else:
            objects.append({
                "object_id": next_object_id,
                "label": label,
                "mean_position": pos,
                "max_confidence": obs["confidence"],
                "frames": [obs["frame"]],
                "observations": [obs],
            })
            next_object_id += 1

    objects_path.parent.mkdir(parents=True, exist_ok=True)

    with open(objects_path, "w", encoding="utf-8") as f:
        json.dump(objects, f, indent=2)

    print(f"Saved object identities to {objects_path}")
    print(f"Unique objects: {len(objects)}")

    for obj in objects:
        print(
            f"Object #{obj['object_id']} | "
            f"{obj['label']} | "
            f"observations={len(obj['observations'])} | "
            f"max_conf={obj['max_confidence']:.2f}"
        )


if __name__ == "__main__":
    main()