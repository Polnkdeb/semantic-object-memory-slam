from pathlib import Path
import json
import csv
import sys
from collections import Counter, defaultdict

import matplotlib.pyplot as plt


def main():
    if len(sys.argv) != 5:
        print(
            "Usage: python src/semantic_map_scene.py "
            "<poses_json> <memory_json> <map_png> <summary_csv>"
        )
        return

    poses_path = Path(sys.argv[1])
    memory_path = Path(sys.argv[2])
    map_path = Path(sys.argv[3])
    csv_path = Path(sys.argv[4])

    map_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(poses_path, "r", encoding="utf-8") as f:
        poses = json.load(f)

    with open(memory_path, "r", encoding="utf-8") as f:
        memory = json.load(f)

    xs = [p["x"] for p in poses]
    zs = [p["z"] for p in poses]

    plt.figure(figsize=(10, 8))
    plt.plot(xs, zs, linewidth=2, label="camera trajectory")
    plt.scatter(xs[0], zs[0], s=80, marker="o", label="start")
    plt.scatter(xs[-1], zs[-1], s=80, marker="x", label="end")

    label_positions = defaultdict(list)

    for obj in memory:
        label = obj["label"]
        pos = obj["camera_position"]
        label_positions[label].append((pos["x"], pos["z"], obj["confidence"]))

    for label, points in label_positions.items():
        px = [p[0] for p in points]
        pz = [p[1] for p in points]

        plt.scatter(px, pz, s=25, alpha=0.7, label=label)

        best_point = max(points, key=lambda p: p[2])
        plt.text(best_point[0], best_point[1], label, fontsize=9)

    plt.xlabel("x")
    plt.ylabel("z")
    plt.title("Semantic Object Memory Map")
    plt.axis("equal")
    plt.grid(True)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(map_path, dpi=250)
    plt.close()

    counter = Counter(obj["label"] for obj in memory)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["object_label", "count"])

        for label, count in counter.most_common():
            writer.writerow([label, count])

    print(f"Saved semantic map to {map_path}")
    print(f"Saved object summary to {csv_path}")


if __name__ == "__main__":
    main()