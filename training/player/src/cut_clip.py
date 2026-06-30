from pathlib import Path
import cv2
import argparse

ROOT = Path(__file__).resolve().parents[1]
VIDEOS_DIR = ROOT / "videos"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="test_30s.mp4")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--duration", type=int, default=30)
    args = parser.parse_args()

    input_path = VIDEOS_DIR / args.input
    output_path = VIDEOS_DIR / args.output

    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(args.start * fps)
    end_frame = int((args.start + args.duration) * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    frame_id = start_frame
    while frame_id < end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        frame_id += 1

    cap.release()
    writer.release()

    print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()