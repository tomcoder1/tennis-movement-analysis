import csv
from collections import defaultdict
from dataclasses import dataclass

from pipeline_utils import DETECTION_COLUMNS, parse_number, resolve_input_path, validate_csv

@dataclass(frozen=True)
class TrackPoint:
    frame: int
    timestamp: float
    x: float
    y: float
    confidence: float
    state: str = "observed"

@dataclass(frozen=True)
class ImageBall:
    frame: int
    timestamp: float
    x: float
    y: float
    confidence: float
    state: str = "observed"

@dataclass(frozen=True)
class PlayerBox:
    player_id: str
    x: float
    y: float
    w: float
    h: float
    confidence: float

@dataclass
class Tracks:
    image_ball: dict
    image_ball_observed: dict
    player_boxes: dict
    ball_court: list
    ball_court_by_frame: dict
    player_court: dict
    valid_homography_frames: set
    net_points: dict
    last_frame: int
    ball_state: dict

def _read(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))

def _valid(row):
    return str(row.get("valid", "")).lower() in {"1", "1.0", "true"}

def _interpolate_image_ball(points, last_frame, max_gap=12, max_step=95.0):
    filled = dict(points)
    state = {frame: "observed" for frame in points}
    frames = sorted(points)
    for left, right in zip(frames, frames[1:]):
        gap = right - left
        if gap <= 1 or gap - 1 > max_gap:
            continue
        a, b = points[left], points[right]
        step = ((b.x - a.x) ** 2 + (b.y - a.y) ** 2) ** 0.5 / gap
        if step > max_step:
            continue
        for frame in range(left + 1, right):
            alpha = (frame - left) / gap
            timestamp = a.timestamp + alpha * (b.timestamp - a.timestamp)
            filled[frame] = ImageBall(
                frame, timestamp,
                a.x + (b.x - a.x) * alpha,
                a.y + (b.y - a.y) * alpha,
                min(a.confidence, b.confidence, 0.55),
                "interpolated",
            )
            state[frame] = "interpolated"
    for frame in range(last_frame + 1):
        state.setdefault(frame, "missing")
    return filled, state

def _interpolate_court_ball(points, max_gap=12, max_step=0.12):
    by_frame = {point.frame: point for point in points}
    frames = sorted(by_frame)
    for left, right in zip(frames, frames[1:]):
        gap = right - left
        if gap <= 1 or gap - 1 > max_gap:
            continue
        a, b = by_frame[left], by_frame[right]
        step = ((b.x - a.x) ** 2 + (b.y - a.y) ** 2) ** 0.5 / gap
        if step > max_step:
            continue
        for frame in range(left + 1, right):
            alpha = (frame - left) / gap
            timestamp = a.timestamp + alpha * (b.timestamp - a.timestamp)
            by_frame[frame] = TrackPoint(
                frame, timestamp,
                a.x + (b.x - a.x) * alpha,
                a.y + (b.y - a.y) * alpha,
                min(a.confidence, b.confidence, 0.55),
                "interpolated",
            )
    return dict(sorted(by_frame.items()))

def load_tracks(output_dir="outputs"):
    paths = {key: resolve_input_path(output_dir, key) for key in (
        "ball", "player", "court", "ball_homography", "player_homography", "homography",
    )}
    validate_csv(paths["ball"], DETECTION_COLUMNS, "predictor")
    validate_csv(paths["player"], DETECTION_COLUMNS, "predictor")
    validate_csv(paths["court"], DETECTION_COLUMNS, "predictor")
    validate_csv(paths["ball_homography"], {"frame_id", "timestamp", "court_x", "court_y", "valid", "confidence"}, "homography")
    validate_csv(paths["player_homography"], {"frame_id", "timestamp", "player_id", "court_x", "court_y", "valid", "confidence"}, "homography")
    validate_csv(paths["homography"], {"frame_id", "valid"}, "homography")

    image_ball_observed, last_frame = {}, 0
    for row in _read(paths["ball"]):
        frame = int(float(row["frame_id"]))
        last_frame = max(last_frame, frame)
        x, y = parse_number(row.get("x")), parse_number(row.get("y"))
        if x is not None and y is not None:
            image_ball_observed[frame] = ImageBall(
                frame, parse_number(row.get("timestamp")) or 0.0, x, y,
                parse_number(row.get("confidence")) or 0.0, "observed",
            )
    image_ball, ball_state = _interpolate_image_ball(image_ball_observed, last_frame)

    player_boxes = defaultdict(list)
    for row in _read(paths["player"]):
        if row.get("object_type") != "player":
            continue
        values = [parse_number(row.get(key)) for key in ("x", "y", "w", "h")]
        if any(value is None for value in values):
            continue
        player_boxes[int(float(row["frame_id"]))].append(PlayerBox(
            row.get("object_id", "").replace("_foot", ""), *values,
            parse_number(row.get("confidence")) or 0.0,
        ))

    ball_court = []
    for row in _read(paths["ball_homography"]):
        x, y = parse_number(row.get("court_x")), parse_number(row.get("court_y"))
        if _valid(row) and x is not None and y is not None:
            ball_court.append(TrackPoint(
                int(float(row["frame_id"])), parse_number(row.get("timestamp")) or 0.0,
                x, y, parse_number(row.get("confidence")) or 0.0, "observed",
            ))
    ball_court_by_frame = _interpolate_court_ball(ball_court)

    player_court = defaultdict(list)
    for row in _read(paths["player_homography"]):
        x, y = parse_number(row.get("court_x")), parse_number(row.get("court_y"))
        if _valid(row) and x is not None and y is not None:
            player_id = row.get("player_id", "").replace("_foot", "")
            player_court[player_id].append(TrackPoint(
                int(float(row["frame_id"])), parse_number(row.get("timestamp")) or 0.0,
                x, y, parse_number(row.get("confidence")) or 0.0,
            ))

    valid_frames = {
        int(float(row["frame_id"])) for row in _read(paths["homography"]) if _valid(row)
    }
    net_points = {}
    for row in _read(paths["court"]):
        if row.get("object_id") != "court_14":
            continue
        x, y = parse_number(row.get("x")), parse_number(row.get("y"))
        if x is not None and y is not None:
            net_points[int(float(row["frame_id"]))] = (x, y)
    return Tracks(
        image_ball, image_ball_observed, dict(player_boxes), sorted(ball_court, key=lambda point: point.frame),
        ball_court_by_frame,
        {key: sorted(value, key=lambda point: point.frame) for key, value in player_court.items()},
        valid_frames, net_points, last_frame, ball_state,
    )
