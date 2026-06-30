from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# -----------------------------
# Settings
# -----------------------------

SOURCE = "test.mp4"
# SOURCE = "rtsp://username:password@camera_ip:554/stream"

MODEL_PATH = Path("model/player_detector_best.pt")

CONF = 0.25
IOU = 0.50
IMG_SIZE = 640
TRACKER = "bytetrack.yaml"

WINDOW_NAME = "Real-time Player Tracking"

PLAYABLE_REGION = np.array(
    [
        [0.08, 0.95],
        [0.92, 0.95],
        [0.76, 0.16],
        [0.24, 0.16],
    ],
    dtype=np.float32,
)


# -----------------------------
# Helpers
# -----------------------------

def get_foot_point(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int(y2)


def get_center_point(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def is_inside_region(region, point):
    return cv2.pointPolygonTest(region, point, False) >= 0


def get_candidates(boxes, region, width, height):
    candidates = []

    if boxes is None:
        return candidates

    for box in boxes:
        if box.id is None:
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        bbox = (int(x1), int(y1), int(x2), int(y2))

        track_id = int(box.id[0])
        conf = float(box.conf[0])

        box_width = x2 - x1
        box_height = y2 - y1

        center_x, center_y = get_center_point(bbox)
        foot = get_foot_point(bbox)

        if box_height < height * 0.03:
            continue

        if box_height > height * 0.90:
            continue

        if box_width > width * 0.40:
            continue

        is_near_player = center_y > height * 0.45

        if is_near_player:
            if not (width * 0.10 < center_x < width * 0.90):
                continue
        else:
            if not is_inside_region(region, foot):
                continue

        candidates.append(
            {
                "track_id": track_id,
                "conf": conf,
                "box": bbox,
                "foot": foot,
                "center_x": center_x,
                "center_y": center_y,
            }
        )

    return candidates


def select_two_players(candidates, width, height):
    near_players = []
    far_players = []

    for player in candidates:
        center_x = player["center_x"]
        center_y = player["center_y"]
        foot_y = player["foot"][1]

        if center_y > height * 0.45:
            if width * 0.12 < center_x < width * 0.88:
                near_players.append(player)
        else:
            if (
                width * 0.18 < center_x < width * 0.78
                and height * 0.07 < center_y < height * 0.46
                and foot_y < height * 0.57
            ):
                far_players.append(player)

    selected = []

    if near_players:
        player_1 = max(near_players, key=lambda p: p["foot"][1])
        selected.append(("Player 1", player_1))

    if far_players:
        player_2 = min(far_players, key=lambda p: abs(p["center_x"] - width * 0.50))
        selected.append(("Player 2", player_2))

    return selected


def draw_player(frame, label, player):
    x1, y1, x2, y2 = player["box"]
    foot_x, foot_y = player["foot"]

    color = (0, 0, 255)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    cv2.circle(frame, (foot_x, foot_y), 5, color, -1)

    text = f"{label} ID:{player['track_id']} {player['conf']:.2f}"

    cv2.putText(
        frame,
        text,
        (x1, max(30, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
    )


# -----------------------------
# Main
# -----------------------------

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Cannot find model: {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))

    cap = cv2.VideoCapture(SOURCE)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {SOURCE}")

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        height, width = frame.shape[:2]

        region = (PLAYABLE_REGION * np.array([width, height])).astype(np.int32)

        results = model.track(
            source=frame,
            tracker=TRACKER,
            conf=CONF,
            iou=IOU,
            imgsz=IMG_SIZE,
            persist=True,
            verbose=False,
        )

        result = results[0]

        candidates = get_candidates(
            boxes=result.boxes,
            region=region,
            width=width,
            height=height,
        )

        selected_players = select_two_players(
            candidates=candidates,
            width=width,
            height=height,
        )

        cv2.polylines(frame, [region], True, (0, 255, 255), 2)

        for label, player in selected_players:
            draw_player(frame, label, player)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()