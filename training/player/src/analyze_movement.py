from pathlib import Path
from ultralytics import YOLO
import cv2
import argparse
import numpy as np
from collections import deque

ROOT = Path(__file__).resolve().parents[1]
VIDEOS_DIR = ROOT / "videos"
OUTPUT_DIR = ROOT / "outputs"

PLAYABLE_REGION = np.array([
    [0.08, 0.95],
    [0.92, 0.95],
    [0.76, 0.16],
    [0.24, 0.16],
], dtype=np.float32)

HISTORY_LENGTH = 10
MOVE_THRESHOLD = 15

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="tennis.mp4")
    parser.add_argument("--start", type=float, default=0, help="Start time in seconds")
    return parser.parse_args()

def get_direction(history):
    if len(history) < 2:
        return "stationary"

    old_x, old_y = history[0]
    new_x, new_y = history[-1]

    dx = new_x - old_x
    dy = new_y - old_y

    if abs(dx) < MOVE_THRESHOLD and abs(dy) < MOVE_THRESHOLD:
        return "stationary"

    if abs(dx) > abs(dy):
        return "moving right" if dx > 0 else "moving left"
    return "moving down" if dy > 0 else "moving up"

def get_speed_level(history):
    if len(history) < 2:
        return "slow"

    old_x, old_y = history[0]
    new_x, new_y = history[-1]

    dist = ((new_x - old_x) ** 2 + (new_y - old_y) ** 2) ** 0.5

    if dist < 20:
        return "slow"
    elif dist < 60:
        return "medium"
    return "fast"

def main():
    args = parse_args()
    video_path = VIDEOS_DIR / args.video

    if not video_path.exists():
        raise FileNotFoundError(f"Cannot find video: {video_path}")

    OUTPUT_DIR.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    start_frame = int(args.start * fps)

    region = (PLAYABLE_REGION * np.array([width, height])).astype(np.int32)

    output_path = OUTPUT_DIR / f"phase3_movement_{video_path.stem}_start_{int(args.start)}s.mp4"

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    model = YOLO("yolo11s.pt")
    track_history = {}

    results = model.track(
        source=str(video_path),
        tracker="bytetrack.yaml",
        classes=[0],
        conf=0.10,
        iou=0.5,
        stream=True,
        persist=True
    )

    for frame_idx, result in enumerate(results):
        if frame_idx < start_frame:
            continue

        frame = result.orig_img.copy()
        cv2.polylines(frame, [region], True, (0, 255, 255), 3)

        selected_players = []
        near_candidates = []
        far_candidates = []

        if result.boxes is not None:
            for box in result.boxes:
                if box.id is None:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                track_id = int(box.id[0])
                conf = float(box.conf[0])

                foot_x = int((x1 + x2) / 2)
                foot_y = int(y2)
                foot_point = (foot_x, foot_y)

                inside = cv2.pointPolygonTest(region, foot_point, False) >= 0
                if not inside:
                    continue

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                item = (track_id, conf, x1, y1, x2, y2, foot_point)

                if center_y > height * 0.50:
                    if width * 0.20 < center_x < width * 0.80:
                        near_candidates.append(item)
                else:
                    if width * 0.22 < center_x < width * 0.70 and height * 0.12 < center_y < height * 0.45:
                        far_candidates.append(item)

            if near_candidates:
                selected_players.append(max(near_candidates, key=lambda x: x[1]))

            if far_candidates:
                selected_players.append(max(far_candidates, key=lambda x: x[1]))

        for track_id, conf, x1, y1, x2, y2, foot_point in selected_players:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            if track_id not in track_history:
                track_history[track_id] = deque(maxlen=HISTORY_LENGTH)

            track_history[track_id].append(foot_point)

            direction = get_direction(track_history[track_id])
            speed = get_speed_level(track_history[track_id])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.circle(frame, foot_point, 6, (0, 0, 255), -1)

            label = f"ID:{track_id} | {direction} | {speed}"

            cv2.putText(
                frame,
                label,
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2
            )

        writer.write(frame)

    writer.release()

    print("Done.")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    main()