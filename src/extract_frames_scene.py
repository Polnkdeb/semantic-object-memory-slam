from pathlib import Path
import sys
import cv2


FRAME_STEP = 10


def main():
    if len(sys.argv) != 3:
        print("Usage: python src/extract_frames_scene.py <video_path> <frames_dir>")
        return

    video_path = Path(sys.argv[1])
    frames_dir = Path(sys.argv[2])
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_id = 0
    saved_id = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_id % FRAME_STEP == 0:
            out_path = frames_dir / f"frame_{saved_id:05d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved_id += 1

        frame_id += 1

    cap.release()

    print(f"Saved {saved_id} frames to {frames_dir}")


if __name__ == "__main__":
    main()