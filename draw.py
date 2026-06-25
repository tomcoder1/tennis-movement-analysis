import os
import csv
import math
import cv2
from collections import defaultdict
from pipeline_utils import (
    DETECTION_COLUMNS, ensure_output_dirs, organized_path, resolve_input_path,
    validate_csv, validate_video,
)

BALL_COLOR = (0, 255, 255)
PLAYER_COLOR = (0, 255, 0)
PLAYER_FOOT_COLOR = (255, 0, 255)
COURT_COLOR = (0, 0, 255)
TEXT_COLOR = (255, 255, 255)
HOMOGRAPHY_PANEL_COLOR = (30, 30, 30)
HOMOGRAPHY_PANEL_BORDER_COLOR = (90, 90, 90)
HOMOGRAPHY_LINE_COLOR = (220, 220, 220)
HOMOGRAPHY_OUTSIDE_LINE_COLOR = (120, 120, 120)
HOMOGRAPHY_BALL_COLOR = (0, 255, 255)
HOMOGRAPHY_PLAYER_COLOR = (255, 0, 255)
HOMOGRAPHY_CLIPPED_COLOR = (0, 120, 255)
SUMMARY_COLOR = (255, 220, 120)

BALL_RADIUS = 6
COURT_RADIUS = 5
FOOT_RADIUS = 5
HOMOGRAPHY_BALL_RADIUS = 5
HOMOGRAPHY_PLAYER_RADIUS = 5

FONT = cv2.FONT_HERSHEY_SIMPLEX

HOMOGRAPHY_VIEW_X_MIN = -0.50
HOMOGRAPHY_VIEW_X_MAX = 1.50
HOMOGRAPHY_VIEW_Y_MIN = -0.85
HOMOGRAPHY_VIEW_Y_MAX = 1.35

def parse_float(value):
    if value is None or value == "":
        return None

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(value):
        return None

    return value

def parse_int(value, default=0):
    if value is None or value == "":
        return default

    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default

def load_csv_by_frame(csv_path):
    detections_by_frame = defaultdict(list)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cannot find CSV: {csv_path}")

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            frame_id = parse_int(row.get("frame_id"))

            det = {
                "frame_id": frame_id,
                "timestamp": parse_float(row.get("timestamp")) or 0.0,
                "source": row.get("source", ""),
                "object_type": row.get("object_type", ""),
                "object_id": row.get("object_id", ""),
                "x": parse_float(row.get("x")),
                "y": parse_float(row.get("y")),
                "w": parse_float(row.get("w")),
                "h": parse_float(row.get("h")),
                "confidence": parse_float(row.get("confidence")) or 0.0,
                "track_id": row.get("track_id", ""),
            }

            detections_by_frame[frame_id].append(det)

    return detections_by_frame

def load_ball_by_frame(csv_path):
    ball_by_frame = {}

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cannot find CSV: {csv_path}")

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            frame_id = parse_int(row.get("frame_id"))

            x = parse_float(row.get("x"))
            y = parse_float(row.get("y"))

            if x is None or y is None:
                continue

            ball_by_frame[frame_id] = {
                "frame_id": frame_id,
                "timestamp": parse_float(row.get("timestamp")) or 0.0,
                "source": row.get("source", "tracknet"),
                "object_type": "ball",
                "object_id": "ball",
                "x": x,
                "y": y,
                "confidence": parse_float(row.get("confidence")) or 0.0,
            }

    return ball_by_frame

def load_ball_homography_by_frame(csv_path):
    ball_by_frame = {}

    if not os.path.exists(csv_path):
        return ball_by_frame

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            frame_id = parse_int(row.get("frame_id"))
            valid = parse_int(row.get("valid"), 0) == 1
            court_x = parse_float(row.get("court_x"))
            court_y = parse_float(row.get("court_y"))

            if not valid or court_x is None or court_y is None:
                continue

            ball_by_frame[frame_id] = {
                "frame_id": frame_id,
                "timestamp": parse_float(row.get("timestamp")) or 0.0,
                "court_x": court_x,
                "court_y": court_y,
                "confidence": parse_float(row.get("confidence")) or 0.0,
            }

    return ball_by_frame

def load_player_homography_by_frame(csv_path):
    players_by_frame = defaultdict(list)

    if not os.path.exists(csv_path):
        return players_by_frame

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            frame_id = parse_int(row.get("frame_id"))
            valid = parse_int(row.get("valid"), 0) == 1
            court_x = parse_float(row.get("court_x"))
            court_y = parse_float(row.get("court_y"))

            if not valid or court_x is None or court_y is None:
                continue

            players_by_frame[frame_id].append({
                "frame_id": frame_id,
                "timestamp": parse_float(row.get("timestamp")) or 0.0,
                "player_id": row.get("player_id", ""),
                "court_x": court_x,
                "court_y": court_y,
                "confidence": parse_float(row.get("confidence")) or 0.0,
                "track_id": row.get("track_id", ""),
            })

    return players_by_frame


def load_scheduled_lines(csv_path, fps):
    lines_by_frame = defaultdict(list)
    if not os.path.exists(csv_path):
        return lines_by_frame

    with open(csv_path, "r", newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("was_skipped", "")).lower() == "true":
                continue
            start = parse_float(row.get("start_timestamp"))
            end = parse_float(row.get("end_timestamp"))
            text = row.get("spoken_text", "").strip()
            if start is None or not text:
                continue
            frame_id = int(round(start * fps))
            lines_by_frame[frame_id].append({"text": text, "end_timestamp": end or start + 1.5})
    return lines_by_frame

def draw_header(frame, frame_id, timestamp):
    cv2.putText(
        frame,
        f"frame: {frame_id}  time: {timestamp:.2f}s",
        (20, 30),
        FONT,
        0.8,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )

def draw_player(frame, det):
    x = int(round(det["x"]))
    y = int(round(det["y"]))
    w = int(round(det["w"]))
    h = int(round(det["h"]))

    x2 = x + w
    y2 = y + h

    label = det["object_id"]

    cv2.rectangle(frame, (x, y), (x2, y2), PLAYER_COLOR, 2)

    cv2.putText(
        frame,
        label,
        (x, max(20, y - 8)),
        FONT,
        0.5,
        PLAYER_COLOR,
        1,
        cv2.LINE_AA,
    )

def draw_player_foot(frame, det):
    x = int(round(det["x"]))
    y = int(round(det["y"]))
    cv2.circle(frame, (x, y), FOOT_RADIUS, PLAYER_FOOT_COLOR, -1)

def draw_court_point(frame, det):
    x = int(round(det["x"]))
    y = int(round(det["y"]))

    label = ""
    cv2.circle(frame, (x, y), COURT_RADIUS, COURT_COLOR, -1)
    cv2.putText(
        frame,
        label,
        (x + 6, y - 6),
        FONT,
        0.4,
        COURT_COLOR,
        1,
        cv2.LINE_AA,
    )

def draw_ball(frame, det):
    x = int(round(det["x"]))
    y = int(round(det["y"]))

    label = "ball"

    cv2.circle(frame, (x, y), BALL_RADIUS, BALL_COLOR, -1)
    cv2.putText(
        frame,
        label,
        (x + 8, y - 8),
        FONT,
        0.5,
        BALL_COLOR,
        1,
        cv2.LINE_AA,
    )


def draw_narration(frame, narration_lines):
    height, width = frame.shape[:2]
    for index, line in enumerate(narration_lines[-2:]):
        description = line.get("text", "")
        if not description:
            continue
        text_width = cv2.getTextSize(description, FONT, 0.62, 2)[0][0]
        x = max(20, (width - text_width) // 2)
        baseline = height - 31 - (len(narration_lines[-2:]) - index - 1) * 39
        cv2.rectangle(frame, (x - 10, baseline - 26),
                      (min(width - 10, x + text_width + 10), baseline + 11),
                      (30, 30, 30), -1)
        cv2.putText(frame, description, (x, baseline), FONT, 0.62,
                    SUMMARY_COLOR, 2, cv2.LINE_AA)

def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))

def homography_panel_layout(frame_width, frame_height):
    margin = 0
    title_h = 50

    panel_w = 260
    panel_h = 455

    x0 = frame_width - panel_w - margin
    y0 = title_h

    return int(x0), int(y0), int(panel_w), int(panel_h)

def point_in_view(court_x, court_y):
    return (
        HOMOGRAPHY_VIEW_X_MIN <= court_x <= HOMOGRAPHY_VIEW_X_MAX
        and HOMOGRAPHY_VIEW_Y_MIN <= court_y <= HOMOGRAPHY_VIEW_Y_MAX
    )

def point_on_court(court_x, court_y):
    return 0.0 <= court_x <= 1.0 and 0.0 <= court_y <= 1.0

def mini_court_point(panel_x, panel_y, panel_w, panel_h, court_x, court_y, clamp_to_view=False):
    if clamp_to_view:
        court_x = clamp(court_x, HOMOGRAPHY_VIEW_X_MIN, HOMOGRAPHY_VIEW_X_MAX)
        court_y = clamp(court_y, HOMOGRAPHY_VIEW_Y_MIN, HOMOGRAPHY_VIEW_Y_MAX)

    x_ratio = (court_x - HOMOGRAPHY_VIEW_X_MIN) / (HOMOGRAPHY_VIEW_X_MAX - HOMOGRAPHY_VIEW_X_MIN)
    y_ratio = (court_y - HOMOGRAPHY_VIEW_Y_MIN) / (HOMOGRAPHY_VIEW_Y_MAX - HOMOGRAPHY_VIEW_Y_MIN)

    x = panel_x + int(round(x_ratio * panel_w))
    y = panel_y + int(round(y_ratio * panel_h))

    return x, y

def draw_panel_background(frame, x0, y0, panel_w, panel_h):
    cv2.rectangle(
        frame,
        (x0 - 10, y0 - 30),
        (x0 + panel_w + 10, y0 + panel_h + 12),
        HOMOGRAPHY_PANEL_COLOR,
        -1,
    )

    cv2.rectangle(
        frame,
        (x0, y0),
        (x0 + panel_w, y0 + panel_h),
        HOMOGRAPHY_PANEL_BORDER_COLOR,
        1,
    )

    cv2.putText(
        frame,
        "homography",
        (x0, y0 - 10),
        FONT,
        0.55,
        TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )

def draw_homography_line(frame, x0, y0, panel_w, panel_h, start, end, color, thickness=1):
    p1 = mini_court_point(x0, y0, panel_w, panel_h, start[0], start[1])
    p2 = mini_court_point(x0, y0, panel_w, panel_h, end[0], end[1])
    cv2.line(frame, p1, p2, color, thickness)

def draw_homography_court_lines(frame, x0, y0, panel_w, panel_h):
    court_lines = [
        ((0.0, 0.0), (1.0, 0.0)),
        ((1.0, 0.0), (1.0, 1.0)),
        ((1.0, 1.0), (0.0, 1.0)),
        ((0.0, 1.0), (0.0, 0.0)),
    ]

    for start, end in court_lines:
        draw_homography_line(frame, x0, y0, panel_w, panel_h, start, end, HOMOGRAPHY_LINE_COLOR, 1)

    singles_left = 0.125
    singles_right = 0.875
    far_service_y = 0.231
    net_y = 0.5
    near_service_y = 0.769

    line_points = [
        ((singles_left, 0.0), (singles_left, 1.0)),
        ((singles_right, 0.0), (singles_right, 1.0)),
        ((singles_left, far_service_y), (singles_right, far_service_y)),
        ((singles_left, net_y), (singles_right, net_y)),
        ((singles_left, near_service_y), (singles_right, near_service_y)),
        ((0.5, far_service_y), (0.5, near_service_y)),
    ]

    for start, end in line_points:
        draw_homography_line(frame, x0, y0, panel_w, panel_h, start, end, HOMOGRAPHY_LINE_COLOR, 1)

    guide_lines = [
        ((HOMOGRAPHY_VIEW_X_MIN, 0.0), (HOMOGRAPHY_VIEW_X_MAX, 0.0)),
        ((HOMOGRAPHY_VIEW_X_MIN, 1.0), (HOMOGRAPHY_VIEW_X_MAX, 1.0)),
        ((0.0, HOMOGRAPHY_VIEW_Y_MIN), (0.0, HOMOGRAPHY_VIEW_Y_MAX)),
        ((1.0, HOMOGRAPHY_VIEW_Y_MIN), (1.0, HOMOGRAPHY_VIEW_Y_MAX)),
    ]

    for start, end in guide_lines:
        draw_homography_line(frame, x0, y0, panel_w, panel_h, start, end, HOMOGRAPHY_OUTSIDE_LINE_COLOR, 1)

def draw_clipped_marker(frame, x, y, color):
    cv2.line(frame, (x - 5, y - 5), (x + 5, y + 5), color, 1)
    cv2.line(frame, (x - 5, y + 5), (x + 5, y - 5), color, 1)

def draw_homography_player(frame, x0, y0, panel_w, panel_h, player):
    court_x = player["court_x"]
    court_y = player["court_y"]
    clipped = not point_in_view(court_x, court_y)
    outside_court = not point_on_court(court_x, court_y)

    px, py = mini_court_point(
        x0,
        y0,
        panel_w,
        panel_h,
        court_x,
        court_y,
        clamp_to_view=True,
    )

    color = HOMOGRAPHY_CLIPPED_COLOR if clipped else HOMOGRAPHY_PLAYER_COLOR
    radius = HOMOGRAPHY_PLAYER_RADIUS + (1 if outside_court else 0)

    cv2.circle(frame, (px, py), radius, color, -1)

    if clipped:
        draw_clipped_marker(frame, px, py, color)

    label = player["player_id"].replace("_foot", "")

    cv2.putText(
        frame,
        label,
        (px + 6, py - 4),
        FONT,
        0.4,
        color,
        1,
        cv2.LINE_AA,
    )

def draw_homography_ball(frame, x0, y0, panel_w, panel_h, ball_h):
    court_x = ball_h["court_x"]
    court_y = ball_h["court_y"]
    clipped = not point_in_view(court_x, court_y)
    outside_court = not point_on_court(court_x, court_y)

    bx, by = mini_court_point(
        x0,
        y0,
        panel_w,
        panel_h,
        court_x,
        court_y,
        clamp_to_view=True,
    )

    color = HOMOGRAPHY_CLIPPED_COLOR if clipped else HOMOGRAPHY_BALL_COLOR
    radius = HOMOGRAPHY_BALL_RADIUS + (1 if outside_court else 0)

    cv2.circle(frame, (bx, by), radius, color, -1)

    if clipped:
        draw_clipped_marker(frame, bx, by, color)

    cv2.putText(
        frame,
        "",
        (bx + 6, by - 4),
        FONT,
        0.4,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_homography_court(frame, ball_h, players_h):
    height, width = frame.shape[:2]
    x0, y0, panel_w, panel_h = homography_panel_layout(width, height)

    if panel_w < 80 or panel_h < 120:
        return

    draw_panel_background(frame, x0, y0, panel_w, panel_h)
    draw_homography_court_lines(frame, x0, y0, panel_w, panel_h)

    for player in players_h:
        draw_homography_player(frame, x0, y0, panel_w, panel_h, player)

    if ball_h is not None:
        draw_homography_ball(frame, x0, y0, panel_w, panel_h, ball_h)

def draw(VIDEO_PATH="test.mp4", output_dir="outputs", max_frames=None):
    validate_video(VIDEO_PATH)
    ensure_output_dirs(output_dir)
    ball_csv = resolve_input_path(output_dir, "ball")
    court_csv = resolve_input_path(output_dir, "court")
    player_csv = resolve_input_path(output_dir, "player")
    for path in (ball_csv, court_csv, player_csv):
        validate_csv(path, DETECTION_COLUMNS, "predictor")

    ball_by_frame = load_ball_by_frame(ball_csv)
    court_by_frame = load_csv_by_frame(court_csv)
    player_by_frame = load_csv_by_frame(player_csv)
    ball_homography_by_frame = load_ball_homography_by_frame(resolve_input_path(output_dir, "ball_homography"))
    player_homography_by_frame = load_player_homography_by_frame(resolve_input_path(output_dir, "player_homography"))
    output_video_path = organized_path(output_dir, "out_video")

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        cap.release()
        raise RuntimeError("Could not read FPS from video.")

    narration_by_frame = load_scheduled_lines(organized_path(output_dir, "scheduled_lines"), fps)

    writer = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output_video_path}")

    frame_id = 0
    recent_narration = []

    while True:
        if max_frames is not None and frame_id >= max_frames:
            break
        ok, frame = cap.read()

        if not ok:
            break

        timestamp = frame_id / fps

        draw_header(frame, frame_id, timestamp)

        for det in court_by_frame.get(frame_id, []):
            if det["object_type"] == "court_point":
                if det["x"] is not None and det["y"] is not None:
                    draw_court_point(frame, det)

        for det in player_by_frame.get(frame_id, []):
            if det["object_type"] == "player":
                if (
                    det["x"] is not None
                    and det["y"] is not None
                    and det["w"] is not None
                    and det["h"] is not None
                ):
                    draw_player(frame, det)

            elif det["object_type"] == "player_foot":
                if det["x"] is not None and det["y"] is not None:
                    draw_player_foot(frame, det)

        current_ball = ball_by_frame.get(frame_id)

        if current_ball is not None:
            draw_ball(frame, current_ball)

        current_narration = narration_by_frame.get(frame_id, [])
        if current_narration:
            recent_narration = current_narration

        draw_homography_court(
            frame=frame,
            ball_h=ball_homography_by_frame.get(frame_id),
            players_h=player_homography_by_frame.get(frame_id, []),
        )

        visible_narration = recent_narration
        if recent_narration and timestamp > max(line["end_timestamp"] for line in recent_narration) + 0.4:
            visible_narration = []
        draw_narration(frame, visible_narration)

        writer.write(frame)

        if frame_id % 100 == 0:
            print(f"Processed draw frame {frame_id}/{total_frames}")

        frame_id += 1

    cap.release()
    writer.release()

    print(f"Saved overlay video to: {output_video_path}")