from pathlib import Path
import argparse
import cv2

ROOT = Path(__file__).resolve().parents[1]
VIDEOS_DIR = ROOT / "videos"
DATASET_DIR = ROOT / "dataset_player"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--step", type=int, default=30)
    return parser.parse_args()


def main():
    args = parse_args()

    video_path = VIDEOS_DIR / args.video

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir = DATASET_DIR / "images" / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_idx % args.step == 0:
            output_path = output_dir / f"{video_path.stem}_{args.split}_{saved_idx:05d}.jpg"
            cv2.imwrite(str(output_path), frame)
            saved_idx += 1

        frame_idx += 1

    cap.release()

    print("Done.")
    print(f"Saved {saved_idx} frames to: {output_dir}")


if __name__ == "__main__":
    main()