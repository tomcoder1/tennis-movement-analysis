import csv
import itertools
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from pipeline_utils import (
    DETECTION_COLUMNS, ensure_output_dirs, organized_path, parse_number,
    resolve_input_path, validate_csv,
)

SINGLES_LEFT = 4.5 / 36.0
SINGLES_RIGHT = 31.5 / 36.0
FAR_SERVICE_Y = 18.0 / 78.0
NET_Y = 39.0 / 78.0
NEAR_SERVICE_Y = 60.0 / 78.0
CENTER_X = 0.5

COURT_TEMPLATE_BY_ID: Dict[str, Tuple[float, float]] = {
    "court_0": (0.0, 0.0),
    "court_1": (1.0, 0.0),
    "court_2": (0.0, 1.0),
    "court_3": (1.0, 1.0),

    "court_4": (SINGLES_LEFT, 0.0),
    "court_5": (SINGLES_LEFT, 1.0),
    "court_6": (SINGLES_RIGHT, 0.0),
    "court_7": (SINGLES_RIGHT, 1.0),

    "court_8": (SINGLES_LEFT, FAR_SERVICE_Y),
    "court_9": (SINGLES_RIGHT, FAR_SERVICE_Y),
    "court_10": (SINGLES_LEFT, NEAR_SERVICE_Y),
    "court_11": (SINGLES_RIGHT, NEAR_SERVICE_Y),

    "court_12": (CENTER_X, FAR_SERVICE_Y),
    "court_13": (CENTER_X, NEAR_SERVICE_Y),
    "court_14": (CENTER_X, NET_Y),
}

MIN_TEMPLATE_POINTS = 4
MIN_QUAD_AREA = 500.0
MIN_SIDE_LENGTH = 20.0
MAX_COURT_POINT_OUTSIDE_RATIO = 0.35
COURT_POINT_MARGIN = 0.35
REUSE_LAST_VALID_H = True

@dataclass
class HomographyResult:
    H: Optional[np.ndarray]
    valid: bool
    method: str
    num_points: int
    reprojection_error: Optional[float]
    timestamp: float

def parse_float(value):
    return parse_number(value)

def parse_int(value, default=0):
    number = parse_number(value)
    if number is None:
        return default
    return int(number)

def load_csv_by_frame(csv_path):
    by_frame = defaultdict(list)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cannot find CSV: {csv_path}")

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            frame_id = parse_int(row.get("frame_id"))

            item = {
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

            by_frame[frame_id].append(item)

    return by_frame

def valid_xy(row):
    x = row.get("x")
    y = row.get("y")
    return x is not None and y is not None and np.isfinite(x) and np.isfinite(y)

def normalize_homography(H):
    if H is None:
        return None

    H = np.asarray(H, dtype=np.float64)

    if H.shape != (3, 3):
        return None

    if not np.all(np.isfinite(H)):
        return None

    if abs(np.linalg.det(H)) < 1e-12:
        return None

    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]

    return H

def transform_points(H, points):
    if H is None or len(points) == 0:
        return np.empty((0, 2), dtype=np.float32)

    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(pts, H)
    return warped.reshape(-1, 2)

def transform_point(H, x, y):
    if H is None or x is None or y is None:
        return None, None

    warped = transform_points(H, [(float(x), float(y))])

    if warped.size == 0:
        return None, None

    court_x = float(warped[0, 0])
    court_y = float(warped[0, 1])

    if not np.isfinite(court_x) or not np.isfinite(court_y):
        return None, None

    return court_x, court_y

def reprojection_error(H, image_points, template_points):
    if H is None or len(image_points) == 0:
        return None

    projected = transform_points(H, image_points)
    target = np.asarray(template_points, dtype=np.float32).reshape(-1, 2)

    if projected.shape != target.shape:
        return None

    errors = np.linalg.norm(projected - target, axis=1)
    return float(np.mean(errors))

def collect_image_points(court_points):
    points = []

    for point in court_points:
        if not valid_xy(point):
            continue
        points.append([float(point["x"]), float(point["y"])])

    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float32)

    arr = np.asarray(points, dtype=np.float32).reshape(-1, 2)

    unique = []
    for pt in arr:
        if not unique:
            unique.append(pt)
            continue

        distances = np.linalg.norm(np.asarray(unique, dtype=np.float32) - pt, axis=1)
        if float(np.min(distances)) >= 2.0:
            unique.append(pt)

    return np.asarray(unique, dtype=np.float32).reshape(-1, 2)


def polygon_area(points):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return abs(float(cv2.contourArea(points)))


def order_quad_for_court(points):
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)

    by_y = pts[np.argsort(pts[:, 1])]
    top = by_y[:2]
    bottom = by_y[2:]

    top = top[np.argsort(top[:, 0])]
    bottom = bottom[np.argsort(bottom[:, 0])]

    top_left, top_right = top[0], top[1]
    bottom_left, bottom_right = bottom[0], bottom[1]

    quad = np.asarray(
        [top_left, top_right, bottom_right, bottom_left],
        dtype=np.float32,
    )

    if polygon_area(quad) < MIN_QUAD_AREA:
        return None

    top_width = np.linalg.norm(top_right - top_left)
    bottom_width = np.linalg.norm(bottom_right - bottom_left)
    left_height = np.linalg.norm(bottom_left - top_left)
    right_height = np.linalg.norm(bottom_right - top_right)

    if min(top_width, bottom_width, left_height, right_height) < MIN_SIDE_LENGTH:
        return None

    width_ratio = max(top_width, bottom_width) / max(1.0, min(top_width, bottom_width))
    height_ratio = max(left_height, right_height) / max(1.0, min(left_height, right_height))

    if width_ratio > 8.0 or height_ratio > 8.0:
        return None

    return quad

def estimate_outer_quad_from_points(points):
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)

    if len(pts) < 4:
        return None

    hull = cv2.convexHull(pts).reshape(-1, 2)

    if len(hull) < 4:
        return None

    best_quad = None
    best_score = -1.0

    for combo in itertools.combinations(hull, 4):
        quad = order_quad_for_court(combo)
        if quad is None:
            continue

        area = polygon_area(quad)

        top_width = np.linalg.norm(quad[1] - quad[0])
        bottom_width = np.linalg.norm(quad[2] - quad[3])
        left_height = np.linalg.norm(quad[3] - quad[0])
        right_height = np.linalg.norm(quad[2] - quad[1])

        width_balance = min(top_width, bottom_width) / max(top_width, bottom_width, 1.0)
        height_balance = min(left_height, right_height) / max(left_height, right_height, 1.0)
        score = area * (0.50 + 0.25 * width_balance + 0.25 * height_balance)

        if score > best_score:
            best_score = score
            best_quad = quad

    if best_quad is not None:
        return best_quad.astype(np.float32)

    sums = pts[:, 0] + pts[:, 1]
    diffs = pts[:, 0] - pts[:, 1]

    rough_quad = np.asarray(
        [
            pts[np.argmin(sums)],
            pts[np.argmax(diffs)],
            pts[np.argmax(sums)],
            pts[np.argmin(diffs)],
        ],
        dtype=np.float32,
    )

    return order_quad_for_court(rough_quad)

def validate_homography_with_court_points(H, image_points):
    if H is None or len(image_points) < 4:
        return False

    warped = transform_points(H, image_points)

    if warped.size == 0 or not np.all(np.isfinite(warped)):
        return False

    lower = -COURT_POINT_MARGIN
    upper = 1.0 + COURT_POINT_MARGIN

    inside = (
        (warped[:, 0] >= lower)
        & (warped[:, 0] <= upper)
        & (warped[:, 1] >= lower)
        & (warped[:, 1] <= upper)
    )

    outside_ratio = 1.0 - (float(np.count_nonzero(inside)) / float(len(warped)))
    return outside_ratio <= MAX_COURT_POINT_OUTSIDE_RATIO

def build_homography_from_template(court_points):
    image_points = []
    template_points = []

    for point in court_points:
        object_id = point.get("object_id", "")

        if object_id not in COURT_TEMPLATE_BY_ID:
            continue

        if not valid_xy(point):
            continue

        image_points.append([float(point["x"]), float(point["y"])])
        template_points.append(COURT_TEMPLATE_BY_ID[object_id])

    timestamp = court_points[0]["timestamp"] if court_points else 0.0

    if len(image_points) < MIN_TEMPLATE_POINTS:
        return HomographyResult(
            H=None,
            valid=False,
            method="id_template_not_enough_points",
            num_points=len(image_points),
            reprojection_error=None,
            timestamp=timestamp,
        )

    image_arr = np.asarray(image_points, dtype=np.float32)
    template_arr = np.asarray(template_points, dtype=np.float32)

    if len(image_arr) == 4:
        H = cv2.getPerspectiveTransform(image_arr, template_arr)
    else:
        H, _ = cv2.findHomography(image_arr, template_arr, cv2.RANSAC, 0.03)

    H = normalize_homography(H)
    err = reprojection_error(H, image_arr, template_arr)
    valid = H is not None and validate_homography_with_court_points(H, image_arr)

    return HomographyResult(
        H=H if valid else None,
        valid=valid,
        method="id_template" if valid else "id_template_invalid_H",
        num_points=len(image_points),
        reprojection_error=err,
        timestamp=timestamp,
    )

def build_homography_from_auto_quad(court_points):
    image_points = collect_image_points(court_points)
    timestamp = court_points[0]["timestamp"] if court_points else 0.0

    if len(image_points) < 4:
        return HomographyResult(
            H=None,
            valid=False,
            method="auto_quad_not_enough_points",
            num_points=len(image_points),
            reprojection_error=None,
            timestamp=timestamp,
        )

    quad = estimate_outer_quad_from_points(image_points)

    if quad is None:
        return HomographyResult(
            H=None,
            valid=False,
            method="auto_quad_failed",
            num_points=len(image_points),
            reprojection_error=None,
            timestamp=timestamp,
        )

    template_quad = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    H = cv2.getPerspectiveTransform(quad.astype(np.float32), template_quad)
    H = normalize_homography(H)
    err = reprojection_error(H, quad, template_quad)
    valid = H is not None and validate_homography_with_court_points(H, image_points)

    return HomographyResult(
        H=H if valid else None,
        valid=valid,
        method="auto_hull_quad" if valid else "auto_hull_quad_invalid_H",
        num_points=len(image_points),
        reprojection_error=err,
        timestamp=timestamp,
    )

def build_homography(court_points):
    if COURT_TEMPLATE_BY_ID:
        result = build_homography_from_template(court_points)
        if result.valid:
            return result

    return build_homography_from_auto_quad(court_points)

def build_homographies_by_frame(court_by_frame, all_frame_ids):
    homography_by_frame: Dict[int, HomographyResult] = {}
    last_valid_result: Optional[HomographyResult] = None

    for frame_id in all_frame_ids:
        court_points = [
            row for row in court_by_frame.get(frame_id, [])
            if row.get("object_type") == "court_point"
        ]

        result = build_homography(court_points)

        if result.valid and result.H is not None:
            last_valid_result = result
            homography_by_frame[frame_id] = result
            continue

        if REUSE_LAST_VALID_H and last_valid_result is not None:
            homography_by_frame[frame_id] = HomographyResult(
                H=last_valid_result.H,
                valid=True,
                method=f"{last_valid_result.method}_reused",
                num_points=result.num_points,
                reprojection_error=last_valid_result.reprojection_error,
                timestamp=result.timestamp,
            )
        else:
            homography_by_frame[frame_id] = result

    return homography_by_frame

def write_homography_csv(homography_by_frame, output_csv):
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_id",
            "timestamp",
            "valid",
            "method",
            "num_points",
            "reprojection_error",
            "h00",
            "h01",
            "h02",
            "h10",
            "h11",
            "h12",
            "h20",
            "h21",
            "h22",
        ])

        for frame_id in sorted(homography_by_frame.keys()):
            result = homography_by_frame[frame_id]
            H = result.H

            if H is None:
                values = [""] * 9
            else:
                values = [float(v) for v in H.reshape(-1)]

            writer.writerow([
                frame_id,
                result.timestamp,
                1 if result.valid and H is not None else 0,
                result.method,
                result.num_points,
                result.reprojection_error if result.reprojection_error is not None else "",
                *values,
            ])

def write_ball_homography_csv(ball_by_frame, homography_by_frame, output_csv):
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_id",
            "timestamp",
            "img_x",
            "img_y",
            "court_x",
            "court_y",
            "valid",
            "confidence",
        ])

        for frame_id in sorted(ball_by_frame.keys()):
            result = homography_by_frame.get(frame_id)
            H = result.H if result is not None and result.valid else None

            for row in ball_by_frame[frame_id]:
                if row.get("object_type") != "ball":
                    continue

                court_x, court_y = transform_point(H, row.get("x"), row.get("y"))
                valid = court_x is not None and court_y is not None

                writer.writerow([
                    frame_id,
                    row.get("timestamp", 0.0),
                    row.get("x") if row.get("x") is not None else "",
                    row.get("y") if row.get("y") is not None else "",
                    court_x if valid else "",
                    court_y if valid else "",
                    1 if valid else 0,
                    row.get("confidence", 0.0),
                ])

def write_player_homography_csv(player_by_frame, homography_by_frame, output_csv):
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_id",
            "timestamp",
            "player_id",
            "img_x",
            "img_y",
            "court_x",
            "court_y",
            "valid",
            "confidence",
            "track_id",
        ])

        for frame_id in sorted(player_by_frame.keys()):
            result = homography_by_frame.get(frame_id)
            H = result.H if result is not None and result.valid else None

            for row in player_by_frame[frame_id]:
                if row.get("object_type") != "player_foot":
                    continue

                court_x, court_y = transform_point(H, row.get("x"), row.get("y"))
                valid = court_x is not None and court_y is not None

                writer.writerow([
                    frame_id,
                    row.get("timestamp", 0.0),
                    row.get("object_id", ""),
                    row.get("x") if row.get("x") is not None else "",
                    row.get("y") if row.get("y") is not None else "",
                    court_x if valid else "",
                    court_y if valid else "",
                    1 if valid else 0,
                    row.get("confidence", 0.0),
                    row.get("track_id", ""),
                ])

def run_homography(output_dir="outputs"):
    ensure_output_dirs(output_dir)
    court_csv = resolve_input_path(output_dir, "court")
    ball_csv = resolve_input_path(output_dir, "ball")
    player_csv = resolve_input_path(output_dir, "player")
    for path in (court_csv, ball_csv, player_csv):
        validate_csv(path, DETECTION_COLUMNS, "predictor")
    court_by_frame = load_csv_by_frame(court_csv)
    ball_by_frame = load_csv_by_frame(ball_csv)
    player_by_frame = load_csv_by_frame(player_csv)

    all_frame_ids = sorted(
        set(court_by_frame.keys())
        | set(ball_by_frame.keys())
        | set(player_by_frame.keys())
    )

    homography_by_frame = build_homographies_by_frame(
        court_by_frame=court_by_frame,
        all_frame_ids=all_frame_ids,
    )

    homography_csv = organized_path(output_dir, "homography")
    ball_h_csv = organized_path(output_dir, "ball_homography")
    player_h_csv = organized_path(output_dir, "player_homography")
    write_homography_csv(homography_by_frame, homography_csv)
    write_ball_homography_csv(ball_by_frame, homography_by_frame, ball_h_csv)
    write_player_homography_csv(player_by_frame, homography_by_frame, player_h_csv)

    valid_count = sum(
        1 for result in homography_by_frame.values()
        if result.valid and result.H is not None
    )

    print("Homography done.")
    print(f"Valid homography frames: {valid_count}/{len(all_frame_ids)}")
    print(f"Saved homography CSV: {homography_csv}")
    print(f"Saved ball homography CSV: {ball_h_csv}")
    print(f"Saved player homography CSV: {player_h_csv}")
