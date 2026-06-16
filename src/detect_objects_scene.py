import json
import os
import sys
from pathlib import Path

from tqdm import tqdm
from ultralytics import YOLO


class YOLODetector:
    def __init__(self, model_name="yolov8n.pt", confidence_threshold=0.4):
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold

    def detect_on_frames(self, frames_dir, output_json_path):
        detections = {}

        image_files = sorted(
            f for f in os.listdir(frames_dir)
            if f.endswith(".jpg")
        )

        total_detections = 0

        for img_file in tqdm(image_files, desc="Detecting"):
            img_path = os.path.join(frames_dir, img_file)
            results = self.model(img_path, verbose=False)[0]

            detections[img_file] = []

            for box in results.boxes:
                cls_id = int(box.cls[0])
                class_name = results.names[cls_id]
                conf = float(box.conf[0])

                if conf < self.confidence_threshold:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                detections[img_file].append({
                    "class": class_name,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2]
                })

                total_detections += 1

        output_json_path = Path(output_json_path)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(detections, f, indent=2)

        print(f"Processed frames: {len(image_files)}")
        print(f"Detected objects: {total_detections}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python src/detect_objects_scene.py <frames_dir> <output_json_path>")
        return

    frames_dir = sys.argv[1]
    output_json_path = sys.argv[2]

    detector = YOLODetector()
    detector.detect_on_frames(frames_dir, output_json_path)


if __name__ == "__main__":
    main()