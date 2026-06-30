from pathlib import Path
from ultralytics import YOLO
import cv2
import argparse
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VIDEOS_DIR = ROOT / "videos"
OUTPUT_DIR = ROOT / "outputs"

PLAYABLE_REGION = np.array([
    [0.04, 0.96],
    [0.96, 0.96],
    [0.86, 0.12],
    [0.14, 0.12],
], dtype=np.float32)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="tennis.mp4")
    return parser.parse_args()

def box_center(x1, y1, x2, y2):
    return ((x1 + x2) / 2, (y1 + y2) / 2)

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

    region = (PLAYABLE_REGION * np.array([width, height])).astype(np.int32)

    output_path = OUTPUT_DIR / f"phase2_filtered_{video_path.stem}.mp4"

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    model = YOLO("yolo11s.pt")

    results = model.track(
        source=str(video_path),
        tracker="bytetrack_custom.yaml",
        classes=[0],
        conf=0.10,
        iou=0.5,
        stream=True,
        persist=True
    )

    for result in results:
        frame = result.orig_img.copy()
        cv2.polylines(frame, [region], True, (0, 255, 255), 3)

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

                box_h = y2 - y1
                box_w = x2 - x1
                area = box_w * box_h

                if conf < 0.15:
                    continue

                center_x, center_y = box_center(x1, y1, x2, y2)

                center_score = -abs(center_x - width / 2)

                score = area * 0.4 + center_score * 0.6

                item = {
                    "score": score,
                    "id": track_id,
                    "conf": conf,
                    "box": (int(x1), int(y1), int(x2), int(y2)),
                    "foot": foot_point
                }

                if center_y > height * 0.50:
                    if width * 0.20 < center_x < width * 0.80:
                        near_candidates.append(item)
                else:
                    if width * 0.35 < center_x < width * 0.65 and height * 0.12 < center_y < height * 0.45:
                        far_candidates.append(item)

        selected = []

        if near_candidates:
            selected.append(max(near_candidates, key=lambda x: x["score"]))

        if far_candidates:
            selected.append(max(far_candidates, key=lambda x: x["score"]))

        for player in selected:
            x1, y1, x2, y2 = player["box"]
            foot = player["foot"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.circle(frame, foot, 6, (0, 0, 255), -1)

            cv2.putText(
                frame,
                f"Player ID:{player['id']} {player['conf']:.2f}",
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        writer.write(frame)

    writer.release()

    print("Done.")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    main()