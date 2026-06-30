from pathlib import Path
from ultralytics import YOLO
import argparse

ROOT = Path(__file__).resolve().parents[1]
VIDEOS_DIR = ROOT / "videos"
OUTPUT_DIR = ROOT / "outputs"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        default="tennis.mp4",
        help="Video filename inside videos folder"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    video_path = VIDEOS_DIR / args.video

    if not video_path.exists():
        raise FileNotFoundError(f"Cannot find video: {video_path}")

    OUTPUT_DIR.mkdir(exist_ok=True)

    model = YOLO("yolo11n.pt")

    output_name = f"phase1_tracking_{video_path.stem}"

    model.track(
        source=str(video_path),
        tracker="bytetrack.yaml",
        classes=[0],
        conf=0.25,
        iou=0.5,
        save=True,
        project=str(OUTPUT_DIR),
        name=output_name,
        exist_ok=True
    )

    print("Done.")
    print(f"Input video: {video_path}")
    print(f"Check output folder: {OUTPUT_DIR / output_name}")

if __name__ == "__main__":
    main()