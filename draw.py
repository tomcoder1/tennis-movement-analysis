import os
import csv
import math
import cv2
from collections import defaultdict

BALL_CSV = "outputs/ball.csv"
COURT_CSV = "outputs/court.csv"
PLAYER_CSV = "outputs/player.csv"
BALL_HOMOGRAPHY_CSV = "outputs/ball_homography.csv"
PLAYER_HOMOGRAPHY_CSV = "outputs/player_homography.csv"
BALL_EVENTS_CSV = "outputs/ball_events.csv"
TENNIS_EVENTS_CSV = "outputs/tennis_events.csv"

OUTPUT_VIDEO_PATH = "outputs/out.mp4"

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
EVENT_COLOR = (0, 165, 255)
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


def load_events_by_frame(csv_path):
    """Load either event CSV; missing optional event files are harmless."""
    events_by_frame = defaultdict(list)
    if not os.path.exists(csv_path):
        return events_by_frame

    with open(csv_path, "r", newline="") as f:
        for row in csv.DictReader(f):
            frame_id = parse_int(row.get("frame_id"))
            row["frame_id"] = frame_id
            for key in ("img_x", "img_y", "court_x", "court_y",
                        "target_court_x", "target_court_y", "confidence"):
                if key in row:
                    row[key] = parse_float(row.get(key))
            events_by_frame[frame_id].append(row)
    return events_by_frame

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


def draw_event_labels(frame, ball_events, tennis_events, summary_events):
    """Render current raw labels plus a short-lived interpreted summary."""
    height, width = frame.shape[:2]
    y = 62
    for event in ball_events:
        player = event.get("player_id", "")
        label = event.get("event_type", "event").replace("_", " ").upper()
        if player:
            label += f" - {player}"
        confidence = event.get("confidence")
        if confidence is not None:
            label += f" ({confidence:.2f})"
        cv2.putText(frame, label, (20, y), FONT, 0.72, EVENT_COLOR, 2, cv2.LINE_AA)
        y += 29

        img_x, img_y = event.get("img_x"), event.get("img_y")
        if img_x is not None and img_y is not None:
            point = (int(round(img_x)), int(round(img_y)))
            cv2.circle(frame, point, 13, EVENT_COLOR, 2)
            cv2.putText(frame, event.get("event_type", ""),
                        (point[0] + 15, max(22, point[1] - 10)),
                        FONT, 0.55, EVENT_COLOR, 2, cv2.LINE_AA)

    # Interpreted labels are shown on their exact frame. The latest description
    # also remains briefly so a human can comfortably read the point summary.
    descriptions = [event.get("description", "") for event in tennis_events if event.get("description")]
    if not descriptions and summary_events:
        descriptions = [summary_events[-1].get("description", "")]
    for description in descriptions[-2:]:
        if not description:
            continue
        text_width = cv2.getTextSize(description, FONT, 0.62, 2)[0][0]
        x = max(20, (width - text_width) // 2)
        cv2.rectangle(frame, (x - 10, height - 57),
                      (min(width - 10, x + text_width + 10), height - 20),
                      (30, 30, 30), -1)
        cv2.putText(frame, description, (x, height - 31), FONT, 0.62,
                    SUMMARY_COLOR, 2, cv2.LINE_AA)

def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def homography_panel_layout(frame_width, frame_height):
    margin = 20
    title_h = 50

    max_panel_w = max(120, frame_width - 2 * margin)
    max_panel_h = max(120, frame_height - title_h - margin)

    preferred_w = min(460, max(300, frame_width // 4))
    preferred_h = int(preferred_w * 1.75)

    panel_w = min(preferred_w, max_panel_w)
    panel_h = min(preferred_h, max_panel_h)

    if panel_h < preferred_h:
        panel_w = max(120, int(panel_h / 1.75))

    x0 = frame_width - panel_w - margin
    y0 = title_h

    if x0 < margin:
        x0 = margin

    if y0 + panel_h + margin > frame_height:
        y0 = max(margin, frame_height - panel_h - margin)

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


def draw_homography_court(frame, ball_h, players_h, ball_events=None, tennis_events=None):
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

    # Current event and its interpreted target are highlighted on the mini-court.
    for event in ball_events or []:
        court_x, court_y = event.get("court_x"), event.get("court_y")
        if court_x is None or court_y is None:
            continue
        ex, ey = mini_court_point(x0, y0, panel_w, panel_h, court_x, court_y, True)
        cv2.circle(frame, (ex, ey), 10, EVENT_COLOR, 2)

    for event in tennis_events or []:
        target_x, target_y = event.get("target_court_x"), event.get("target_court_y")
        if target_x is None or target_y is None:
            continue
        tx, ty = mini_court_point(x0, y0, panel_w, panel_h, target_x, target_y, True)
        cv2.drawMarker(frame, (tx, ty), SUMMARY_COLOR, cv2.MARKER_CROSS, 14, 2)

def draw(VIDEO_PATH):
    os.makedirs("outputs", exist_ok=True)

    ball_by_frame = load_ball_by_frame(BALL_CSV)
    court_by_frame = load_csv_by_frame(COURT_CSV)
    player_by_frame = load_csv_by_frame(PLAYER_CSV)
    ball_homography_by_frame = load_ball_homography_by_frame(BALL_HOMOGRAPHY_CSV)
    player_homography_by_frame = load_player_homography_by_frame(PLAYER_HOMOGRAPHY_CSV)
    ball_events_by_frame = load_events_by_frame(BALL_EVENTS_CSV)
    tennis_events_by_frame = load_events_by_frame(TENNIS_EVENTS_CSV)

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

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {OUTPUT_VIDEO_PATH}")

    frame_id = 0
    recent_summary_events = []

    while True:
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

        current_ball_events = ball_events_by_frame.get(frame_id, [])
        current_tennis_events = tennis_events_by_frame.get(frame_id, [])
        summary_candidates = [
            event for event in current_tennis_events
            if event.get("event_type") in ("serve", "return", "rally_hit", "bounce_in",
                                            "bounce_out", "out", "miss", "failed_return", "point_won")
        ]
        if summary_candidates:
            recent_summary_events = summary_candidates

        draw_homography_court(
            frame=frame,
            ball_h=ball_homography_by_frame.get(frame_id),
            players_h=player_homography_by_frame.get(frame_id, []),
            ball_events=current_ball_events,
            tennis_events=current_tennis_events,
        )

        # Keep summaries for 1.5 seconds, but raw labels only on the event frame.
        recent_frame = recent_summary_events[-1]["frame_id"] if recent_summary_events else -10_000
        visible_summary = recent_summary_events if frame_id - recent_frame <= int(1.5 * fps) else []
        draw_event_labels(frame, current_ball_events, current_tennis_events, visible_summary)

        writer.write(frame)

        if frame_id % 100 == 0:
            print(f"Processed draw frame {frame_id}/{total_frames}")

        frame_id += 1

    cap.release()
    writer.release()

    print(f"Saved overlay video to: {OUTPUT_VIDEO_PATH}")
