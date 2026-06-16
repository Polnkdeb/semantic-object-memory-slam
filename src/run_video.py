from pathlib import Path
import subprocess
import sys


DATA_DIR = Path("data")
RUNS_DIR = Path("runs")
VIDEO_EXTENSIONS = [".mp4", ".MP4", ".mov", ".MOV", ".avi", ".mkv"]


def find_videos():
    videos = []

    for path in DATA_DIR.iterdir():
        if path.suffix in VIDEO_EXTENSIONS:
            videos.append(path)

    return sorted(videos)


def run_command(command):
    print("\n" + "=" * 60)
    print(" ".join(command))
    print("=" * 60)

    result = subprocess.run(command)

    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(command)}")


def process_video(video_path):
    scene_name = video_path.stem
    scene_dir = RUNS_DIR / scene_name
    frames_dir = scene_dir / "frames"
    outputs_dir = scene_dir / "outputs"

    frames_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing video: {video_path}")
    print(f"Scene folder: {scene_dir}")

    run_command([
        sys.executable,
        "src/extract_frames_scene.py",
        str(video_path),
        str(frames_dir),
    ])

    run_command([
        sys.executable,
        "src/detect_objects_scene.py",
        str(frames_dir),
        str(outputs_dir / "detections.json"),
    ])

    run_command([
        sys.executable,
        "src/visual_odometry_scene.py",
        str(frames_dir),
        str(outputs_dir / "poses.json"),
    ])

    run_command([
        sys.executable,
        "src/build_memory_scene.py",
        str(outputs_dir / "detections.json"),
        str(outputs_dir / "poses.json"),
        str(outputs_dir / "memory.json"),
    ])

    run_command([
        sys.executable,
        "src/object_identity_scene.py",
        str(outputs_dir / "memory.json"),
        str(outputs_dir / "objects.json"),
    ])

    run_command([
        sys.executable,
        "src/semantic_map_scene.py",
        str(outputs_dir / "poses.json"),
        str(outputs_dir / "memory.json"),
        str(outputs_dir / "semantic_map.png"),
        str(outputs_dir / "object_summary.csv"),
    ])

    print(f"\nDone for scene: {scene_name}")
    print(f"Results saved to: {outputs_dir}")


def main():
    videos = find_videos()

    if not videos:
        print("No videos found in data/")
        return

    print(f"Found {len(videos)} video(s):")
    for video in videos:
        print(f"- {video}")

    for video in videos:
        process_video(video)


if __name__ == "__main__":
    main()