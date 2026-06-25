import os
import csv
import cv2
import numpy as np
import torch

from scipy.spatial import distance
try:
    from ultralytics import YOLO
except ImportError:  # Allows CSV-only stages to run without model dependencies.
    YOLO = None

from model import BallTracker, CourtTracker
from read_video import read_video
from pipeline_utils import ensure_output_dirs, organized_path, require_files, validate_video

BALL_WEIGHTS = "model/model_best.pt"
COURT_WEIGHTS = "model/Court_detect_model.pth"
PLAYER_WEIGHTS = "model/player_detector_best.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INPUT_H = 360
INPUT_W = 640

COURT_DETECTION_INTERVAL_FRAMES = 30

MAX_FRAMES = None

PLAYER_CONF = 0.25
PLAYER_IOU = 0.50
PLAYER_IMG_SIZE = 640
PLAYER_TRACKER = "bytetrack.yaml"

BALL_HEATMAP_THRESHOLD = 127
BALL_MIN_COMPONENT_AREA = 1
BALL_MAX_COMPONENT_AREA = 250
BALL_MAX_COMPONENT_SIDE = 30

BALL_MIN_ACCEPT_SCORE = 80
BALL_MIN_START_SCORE = 150
BALL_MIN_RESTART_SCORE = 165

BALL_MAX_DIST_PER_FRAME = 100
BALL_MAX_DIST_CAP = 220
BALL_REACQUIRE_AFTER_FRAMES = 8
BALL_RESET_AFTER_MISSES = 18
BALL_RESTART_CONFIRM_MAX_GAP = 3
BALL_RESTART_CONFIRM_MAX_DIST = 140
BALL_RESTART_CONFIRM_MIN_MOVE = 5
BALL_HISTORY_SIZE = 5

BALL_MIN_RUN_LENGTH = 5
BALL_INTERPOLATE_MAX_STEP = 80

PLAYABLE_REGION = np.array(
    [
        [0.08, 0.95],
        [0.92, 0.95],
        [0.76, 0.16],
        [0.24, 0.16],
    ],
    dtype=np.float32,
)

SINGLES_LEFT = 4.5 / 36.0
SINGLES_RIGHT = 31.5 / 36.0
FAR_SERVICE_Y = 18.0 / 78.0
NET_Y = 39.0 / 78.0
NEAR_SERVICE_Y = 60.0 / 78.0

COURT_TEMPLATE_BY_ID = {
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
    "court_12": (0.5, FAR_SERVICE_Y),
    "court_13": (0.5, NEAR_SERVICE_Y),
    "court_14": (0.5, NET_Y),
}

PLAYER_COURT_X_MIN = -0.10
PLAYER_COURT_X_MAX = 1.10
PLAYER_COURT_Y_MIN = -0.30
PLAYER_COURT_Y_MAX = 1.30
PLAYER_SIDE_SPLIT_Y = 0.50

def load_pytorch_weights(model, weights_path, device):
    checkpoint = torch.load(weights_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    clean_state = {}

    for key, value in checkpoint.items():
        if key.startswith("module."):
            key = key[len("module."):]
        clean_state[key] = value

    model.load_state_dict(clean_state, strict=True)
    model.to(device)

    return model

def write_csv_header(writer):
    writer.writerow([
        "frame_id",
        "timestamp",
        "source",
        "object_type",
        "object_id",
        "x",
        "y",
        "w",
        "h",
        "confidence",
        "track_id",
    ])

def is_valid_point(point):
    return point is not None and point[0] is not None and point[1] is not None

def get_center_point(x1, y1, x2, y2):
    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
    )

def get_foot_point(x1, y1, x2, y2):
    return (
        (x1 + x2) / 2.0,
        y2,
    )

def is_inside_region(point, width, height):
    x, y = point

    polygon = PLAYABLE_REGION.copy()
    polygon[:, 0] *= width
    polygon[:, 1] *= height

    return cv2.pointPolygonTest(
        polygon.astype(np.float32),
        (float(x), float(y)),
        False,
    ) >= 0

def get_candidates(results, frame_width, frame_height):
    candidates = []

    if results is None or len(results) == 0:
        return candidates

    boxes = results[0].boxes

    if boxes is None:
        return candidates

    for box in boxes:
        if box.id is None:
            continue

        x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy()

        box_w = x2 - x1
        box_h = y2 - y1

        if box_h < 30:
            continue

        if box_h > frame_height * 0.85:
            continue

        if box_w > frame_width * 0.45:
            continue

        center = get_center_point(x1, y1, x2, y2)
        foot = get_foot_point(x1, y1, x2, y2)

        center_x, center_y = center

        if center_y > frame_height * 0.45:
            if center_x < frame_width * 0.10 or center_x > frame_width * 0.90:
                continue
        else:
            if not is_inside_region(foot, frame_width, frame_height):
                continue

        track_id = int(box.id.item())
        conf = float(box.conf.item()) if box.conf is not None else 0.0

        candidates.append({
            "track_id": track_id,
            "bbox": (float(x1), float(y1), float(x2), float(y2)),
            "center": center,
            "foot": foot,
            "confidence": conf,
        })

    return candidates

def build_player_filter_homography(court_points):
    image_points = []
    template_points = []

    if not court_points:
        return None

    for point in court_points:
        point_id = point.get("id", point.get("object_id", ""))

        if point_id not in COURT_TEMPLATE_BY_ID:
            continue

        x = point.get("x")
        y = point.get("y")

        if x is None or y is None:
            continue

        if not np.isfinite(float(x)) or not np.isfinite(float(y)):
            continue

        image_points.append([float(x), float(y)])
        template_points.append(COURT_TEMPLATE_BY_ID[point_id])

    if len(image_points) < 4:
        return None

    image_arr = np.asarray(image_points, dtype=np.float32)
    template_arr = np.asarray(template_points, dtype=np.float32)

    if len(image_arr) == 4:
        H = cv2.getPerspectiveTransform(image_arr, template_arr)
    else:
        H, _ = cv2.findHomography(image_arr, template_arr, cv2.RANSAC, 0.03)

    if H is None:
        return None

    H = np.asarray(H, dtype=np.float64)

    if H.shape != (3, 3):
        return None

    if not np.all(np.isfinite(H)):
        return None

    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]

    return H

def project_point_to_court(H, point):
    if H is None or point is None:
        return None

    pts = np.asarray([[point]], dtype=np.float32)
    warped = cv2.perspectiveTransform(pts, H).reshape(-1, 2)

    if warped.size == 0:
        return None

    court_x = float(warped[0, 0])
    court_y = float(warped[0, 1])

    if not np.isfinite(court_x) or not np.isfinite(court_y):
        return None

    return court_x, court_y

def is_valid_player_court_point(court_point):
    if court_point is None:
        return False

    court_x, court_y = court_point

    return (
        PLAYER_COURT_X_MIN <= court_x <= PLAYER_COURT_X_MAX
        and PLAYER_COURT_Y_MIN <= court_y <= PLAYER_COURT_Y_MAX
    )

def attach_court_positions_to_candidates(candidates, H):
    if H is None:
        return candidates

    filtered = []

    for candidate in candidates:
        court_foot = project_point_to_court(H, candidate.get("foot"))

        if not is_valid_player_court_point(court_foot):
            continue

        item = candidate.copy()
        item["court_foot"] = court_foot
        filtered.append(item)

    return filtered

def select_two_players(candidates, frame_width, frame_height):
    selected = {}

    court_candidates = [
        candidate for candidate in candidates
        if candidate.get("court_foot") is not None
    ]

    if court_candidates:
        near_candidates = []
        far_candidates = []

        for candidate in court_candidates:
            court_x, court_y = candidate["court_foot"]

            if court_y >= PLAYER_SIDE_SPLIT_Y:
                near_candidates.append(candidate)
            else:
                far_candidates.append(candidate)

        if near_candidates:
            near = max(
                near_candidates,
                key=lambda c: (c["court_foot"][1], c["confidence"]),
            )
            selected["player_1"] = near

        if far_candidates:
            far = min(
                far_candidates,
                key=lambda c: (c["court_foot"][1], -c["confidence"]),
            )
            selected["player_2"] = far

        return selected

    near_candidates = []
    far_candidates = []

    for candidate in candidates:
        center_x, center_y = candidate["center"]
        foot_x, foot_y = candidate["foot"]

        if center_y > frame_height * 0.45:
            if frame_width * 0.12 <= center_x <= frame_width * 0.88:
                near_candidates.append(candidate)
        else:
            if foot_y < frame_height * 0.57:
                far_candidates.append(candidate)

    if near_candidates:
        near = max(near_candidates, key=lambda c: c["foot"][1])
        selected["player_1"] = near

    if far_candidates:
        far = min(
            far_candidates,
            key=lambda c: abs(c["center"][0] - frame_width * 0.50),
        )
        selected["player_2"] = far

    return selected

def preprocess_court_frame(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (INPUT_W, INPUT_H))

    frame_rgb = frame_rgb.astype(np.float32) / 255.0
    frame_rgb = np.rollaxis(frame_rgb, 2, 0)

    tensor = torch.from_numpy(frame_rgb).float().unsqueeze(0).to(DEVICE)

    return tensor

def extract_court_keypoints(court_out, original_w, original_h):
    if isinstance(court_out, (tuple, list)):
        court_out = court_out[0]

    heatmaps = court_out.squeeze(0).detach().cpu().numpy()

    points = []

    for idx in range(heatmaps.shape[0]):
        heatmap = heatmaps[idx]

        y_model, x_model = np.unravel_index(
            np.argmax(heatmap),
            heatmap.shape,
        )

        confidence = float(heatmap[y_model, x_model])

        x_original = x_model * original_w / INPUT_W
        y_original = y_model * original_h / INPUT_H

        points.append({
            "id": f"court_{idx}",
            "x": float(x_original),
            "y": float(y_original),
            "confidence": confidence,
        })

    return points

def preprocess_ball_window(tracknet_window_bgr):
    resized_frames = []

    for frame_bgr in tracknet_window_bgr:
        resized = cv2.resize(frame_bgr, (INPUT_W, INPUT_H))
        resized_frames.append(resized)

    stacked = np.concatenate(resized_frames, axis=2)
    stacked = stacked.astype(np.float32) / 255.0
    stacked = np.rollaxis(stacked, 2, 0)
    stacked = np.expand_dims(stacked, axis=0)

    return torch.from_numpy(stacked).float().to(DEVICE)

def model_output_to_heatmap(model_out):
    if isinstance(model_out, (tuple, list)):
        model_out = model_out[0]

    if model_out.dim() == 4:
        if model_out.shape[1] == 1:
            heatmap = model_out[0, 0]
        else:
            heatmap = model_out.argmax(dim=1)[0]
    elif model_out.dim() == 3:
        if model_out.shape[1] == 256:
            heatmap = model_out.argmax(dim=1)[0].reshape(INPUT_H, INPUT_W)
        elif model_out.shape[0] == 1 and tuple(model_out.shape[1:]) == (INPUT_H, INPUT_W):
            heatmap = model_out[0]
        else:
            heatmap = model_out.reshape(INPUT_H, INPUT_W)
    elif model_out.dim() == 2:
        heatmap = model_out.reshape(INPUT_H, INPUT_W)
    else:
        raise ValueError(f"Unexpected TrackNet output shape: {tuple(model_out.shape)}")

    heatmap = heatmap.detach().float().cpu().numpy()

    if float(np.max(heatmap)) <= 1.0:
        heatmap = heatmap * 255.0

    return np.clip(heatmap, 0, 255).astype(np.uint8)

def candidate_quality(candidate):
    score = float(candidate.get("score", 0.0))
    area = float(candidate.get("area", 0.0))
    width = max(float(candidate.get("w", 1.0)), 1.0)
    height = max(float(candidate.get("h", 1.0)), 1.0)

    area_penalty = abs(area - 25.0) * 0.12
    aspect_penalty = abs(width - height) * 0.25
    size_penalty = max(0.0, max(width, height) - 20.0) * 0.20

    return score - area_penalty - aspect_penalty - size_penalty

def heatmap_to_ball_candidates(heatmap):
    _, binary = cv2.threshold(
        heatmap,
        BALL_HEATMAP_THRESHOLD,
        255,
        cv2.THRESH_BINARY,
    )

    candidates = []

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])

        if area < BALL_MIN_COMPONENT_AREA:
            continue
        if area > BALL_MAX_COMPONENT_AREA:
            continue
        if w > BALL_MAX_COMPONENT_SIDE or h > BALL_MAX_COMPONENT_SIDE:
            continue

        mask = labels == label_id
        ys, xs = np.where(mask)
        values = heatmap[mask].astype(np.float32)

        if values.size > 0 and float(values.sum()) > 0.0:
            cx = float(np.average(xs, weights=values))
            cy = float(np.average(ys, weights=values))
            score = float(values.max())
        else:
            cx, cy = centroids[label_id]
            score = 0.0

        candidates.append({
            "x": cx,
            "y": cy,
            "score": score,
            "area": float(area),
            "w": float(w),
            "h": float(h),
            "source": "component",
        })

    if not candidates:
        circles = cv2.HoughCircles(
            binary,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=4,
            param1=50,
            param2=2,
            minRadius=2,
            maxRadius=7,
        )

        if circles is not None:
            for cx, cy, radius in circles[0]:
                ix = int(round(cx))
                iy = int(round(cy))

                if ix < 0 or ix >= INPUT_W or iy < 0 or iy >= INPUT_H:
                    continue

                candidates.append({
                    "x": float(cx),
                    "y": float(cy),
                    "score": float(heatmap[iy, ix]),
                    "area": float(np.pi * radius * radius),
                    "w": float(radius * 2.0),
                    "h": float(radius * 2.0),
                    "source": "hough",
                })

    candidates.sort(key=candidate_quality, reverse=True)
    return candidates

def scale_ball_candidates(candidates, original_w, original_h):
    scale_x = original_w / INPUT_W
    scale_y = original_h / INPUT_H
    scaled = []

    for candidate in candidates:
        item = candidate.copy()
        item["x"] = float(candidate["x"] * scale_x)
        item["y"] = float(candidate["y"] * scale_y)
        item["w"] = float(candidate["w"] * scale_x)
        item["h"] = float(candidate["h"] * scale_y)
        scaled.append(item)

    return scaled

def infer_ball_candidates_from_packet(packet, model, original_w, original_h):
    if packet.tracknet_window_bgr is None:
        return []

    inp = preprocess_ball_window(packet.tracknet_window_bgr)

    with torch.inference_mode():
        out = model(inp)

    heatmap = model_output_to_heatmap(out)
    candidates = heatmap_to_ball_candidates(heatmap)

    return scale_ball_candidates(candidates, original_w, original_h)

def point_distance(a, b):
    return distance.euclidean(a, b)

def candidate_point(candidate):
    return (float(candidate["x"]), float(candidate["y"]))

def allowed_motion_distance(frame_gap):
    frame_gap = max(1, int(frame_gap))
    return min(BALL_MAX_DIST_PER_FRAME * frame_gap, BALL_MAX_DIST_CAP)

class BallMotionTracker:
    def __init__(self):
        self.history = []
        self.missed = 0
        self.pending_restart = None

    def reset(self):
        self.history.clear()
        self.missed = 0
        self.pending_restart = None

    def update(self, frame_id, candidates):
        candidates = [
            candidate for candidate in candidates
            if float(candidate.get("score", 0.0)) >= BALL_MIN_ACCEPT_SCORE
        ]

        if not candidates:
            return self._mark_missing()

        if not self.history:
            return self._start_or_confirm(frame_id, candidates, BALL_MIN_START_SCORE)

        selected = self._match_existing_track(frame_id, candidates)

        if selected is not None:
            return self._accept(frame_id, selected)

        self.missed += 1

        if self.missed >= BALL_RESET_AFTER_MISSES:
            self.history.clear()

        if self.missed >= BALL_REACQUIRE_AFTER_FRAMES:
            return self._start_or_confirm(frame_id, candidates, BALL_MIN_RESTART_SCORE)

        return None

    def _accept(self, frame_id, candidate):
        point = candidate_point(candidate)
        self.history.append((int(frame_id), point))
        self.history = self.history[-BALL_HISTORY_SIZE:]
        self.missed = 0
        self.pending_restart = None
        return point

    def _mark_missing(self):
        self.missed += 1

        if self.missed >= BALL_RESET_AFTER_MISSES:
            self.history.clear()
            self.pending_restart = None

        return None

    def _start_or_confirm(self, frame_id, candidates, min_score):
        strong_candidates = [
            candidate for candidate in candidates
            if float(candidate.get("score", 0.0)) >= min_score
        ]

        if not strong_candidates:
            self.pending_restart = None
            return None

        best = max(strong_candidates, key=candidate_quality)

        if self.pending_restart is not None:
            prev_frame_id, prev_candidate = self.pending_restart
            frame_gap = frame_id - prev_frame_id

            if 1 <= frame_gap <= BALL_RESTART_CONFIRM_MAX_GAP:
                move = point_distance(candidate_point(best), candidate_point(prev_candidate))
                max_move = min(
                    BALL_RESTART_CONFIRM_MAX_DIST * frame_gap,
                    BALL_MAX_DIST_CAP,
                )

                if BALL_RESTART_CONFIRM_MIN_MOVE <= move <= max_move:
                    self.history.clear()
                    return self._accept(frame_id, best)

        self.pending_restart = (int(frame_id), best)
        return None

    def _match_existing_track(self, frame_id, candidates):
        predicted = self._predict(frame_id)
        last_frame_id, last_point = self.history[-1]
        frame_gap = max(1, frame_id - last_frame_id)
        gate = allowed_motion_distance(frame_gap)

        scored = []

        for candidate in candidates:
            point = candidate_point(candidate)
            distance_to_prediction = point_distance(point, predicted)

            if distance_to_prediction > gate:
                continue

            cost = self._match_cost(candidate, point, predicted, last_point, frame_gap)
            scored.append((cost, candidate))

        if not scored:
            return None

        scored.sort(key=lambda item: item[0])
        return scored[0][1]

    def _predict(self, frame_id):
        last_frame_id, last_point = self.history[-1]

        if len(self.history) < 2:
            return last_point

        velocities = []

        for idx in range(1, len(self.history)):
            prev_frame_id, prev_point = self.history[idx - 1]
            curr_frame_id, curr_point = self.history[idx]
            dt = max(1, curr_frame_id - prev_frame_id)
            vx = (curr_point[0] - prev_point[0]) / dt
            vy = (curr_point[1] - prev_point[1]) / dt
            velocities.append((vx, vy))

        weights = np.arange(1, len(velocities) + 1, dtype=np.float32)
        vx = float(np.average([v[0] for v in velocities], weights=weights))
        vy = float(np.average([v[1] for v in velocities], weights=weights))

        dt = max(1, frame_id - last_frame_id)
        return (
            last_point[0] + vx * dt,
            last_point[1] + vy * dt,
        )

    def _match_cost(self, candidate, point, predicted, last_point, frame_gap):
        distance_to_prediction = point_distance(point, predicted)
        quality = candidate_quality(candidate)

        cost = distance_to_prediction - quality * 0.20

        if len(self.history) >= 2:
            prev_frame_id, prev_point = self.history[-2]
            last_frame_id, _ = self.history[-1]
            prev_dt = max(1, last_frame_id - prev_frame_id)
            curr_dt = max(1, frame_gap)

            prev_vx = (last_point[0] - prev_point[0]) / prev_dt
            prev_vy = (last_point[1] - prev_point[1]) / prev_dt
            curr_vx = (point[0] - last_point[0]) / curr_dt
            curr_vy = (point[1] - last_point[1]) / curr_dt

            speed_change = abs(
                np.hypot(curr_vx, curr_vy) - np.hypot(prev_vx, prev_vy)
            )
            acceleration_penalty = min(speed_change, BALL_MAX_DIST_PER_FRAME) * 0.25
            cost += acceleration_penalty

        return cost

def infer_ball_from_packet(packet, model, original_w, original_h, tracker):
    candidates = infer_ball_candidates_from_packet(
        packet=packet,
        model=model,
        original_w=original_w,
        original_h=original_h,
    )

    point = tracker.update(packet.frame_id, candidates)

    if point is None:
        return None, None

    return int(round(point[0])), int(round(point[1]))

def previous_valid_point(track, index):
    for i in range(index - 1, -1, -1):
        if is_valid_point(track[i]):
            return i, track[i]
    return None, None

def next_valid_point(track, index):
    for i in range(index + 1, len(track)):
        if is_valid_point(track[i]):
            return i, track[i]
    return None, None

def remove_motion_outliers(ball_track, max_dist_per_frame=BALL_MAX_DIST_PER_FRAME):
    cleaned = list(ball_track)

    for _ in range(3):
        to_remove = []

        for i, point in enumerate(cleaned):
            if not is_valid_point(point):
                continue

            prev_i, prev_point = previous_valid_point(cleaned, i)
            next_i, next_point = next_valid_point(cleaned, i)

            if prev_point is None and next_point is None:
                continue

            if prev_point is not None:
                prev_allowed = min(max_dist_per_frame * (i - prev_i), BALL_MAX_DIST_CAP)
                bad_from_prev = point_distance(point, prev_point) > prev_allowed
            else:
                bad_from_prev = False

            if next_point is not None:
                next_allowed = min(max_dist_per_frame * (next_i - i), BALL_MAX_DIST_CAP)
                bad_to_next = point_distance(point, next_point) > next_allowed
            else:
                bad_to_next = False

            if prev_point is not None and next_point is not None:
                span = max(1, next_i - prev_i)
                alpha = (i - prev_i) / span
                expected = (
                    prev_point[0] + (next_point[0] - prev_point[0]) * alpha,
                    prev_point[1] + (next_point[1] - prev_point[1]) * alpha,
                )
                bridge_step = point_distance(prev_point, next_point) / span
                bridge_error = point_distance(point, expected)
                bad_bridge = bridge_error > max(BALL_INTERPOLATE_MAX_STEP, bridge_step * 1.75)

                if (bad_from_prev and bad_to_next) or bad_bridge:
                    to_remove.append(i)
            elif bad_from_prev or bad_to_next:
                to_remove.append(i)

        if not to_remove:
            break

        for i in to_remove:
            cleaned[i] = (None, None)

    return cleaned

def remove_short_runs(ball_track, min_run_length=BALL_MIN_RUN_LENGTH):
    cleaned = list(ball_track)
    i = 0

    while i < len(cleaned):
        if not is_valid_point(cleaned[i]):
            i += 1
            continue

        start = i

        while i < len(cleaned) and is_valid_point(cleaned[i]):
            i += 1

        end = i
        run_len = end - start
        has_left_gap = start == 0 or not is_valid_point(cleaned[start - 1])
        has_right_gap = end == len(cleaned) or not is_valid_point(cleaned[end])

        if run_len < min_run_length and has_left_gap and has_right_gap:
            for idx in range(start, end):
                cleaned[idx] = (None, None)

    return cleaned

def save_ball_csv(ball_track, path_output_csv, fps):
    os.makedirs(os.path.dirname(path_output_csv), exist_ok=True)

    with open(path_output_csv, "w", newline="") as f:
        writer = csv.writer(f)

        write_csv_header(writer)

        for frame_id, point in enumerate(ball_track):
            x, y = point

            writer.writerow([
                frame_id,
                frame_id / fps,
                "tracknet",
                "ball",
                "ball",
                int(round(x)) if x is not None else "",
                int(round(y)) if y is not None else "",
                "",
                "",
                1.0 if x is not None and y is not None else 0.0,
                "",
            ])

def predictor(VIDEO_PATH="test.mp4", output_dir="outputs", max_frames=None):
    validate_video(VIDEO_PATH)
    require_files([BALL_WEIGHTS, COURT_WEIGHTS, PLAYER_WEIGHTS], "model weights")
    ensure_output_dirs(output_dir)
    ball_csv = organized_path(output_dir, "ball")
    court_csv = organized_path(output_dir, "court")
    player_csv = organized_path(output_dir, "player")

    print("Device:", DEVICE)

    ball_model = BallTracker(out_channels=256)
    ball_model = load_pytorch_weights(ball_model, BALL_WEIGHTS, DEVICE)
    ball_model.eval()
    print("Ball model loaded")

    court_model = CourtTracker(out_channels=15)
    court_model = load_pytorch_weights(court_model, COURT_WEIGHTS, DEVICE)
    court_model.eval()
    print("Court model loaded")

    if YOLO is None:
        raise ImportError("ultralytics is required for player detection. Install it with: pip install ultralytics")

    player_model = YOLO(PLAYER_WEIGHTS)
    print("Player model loaded")

    ball_track = []
    ball_motion_tracker = BallMotionTracker()
    last_court_points = None
    fps = None

    with open(court_csv, "w", newline="") as court_file, \
         open(player_csv, "w", newline="") as player_file:

        court_writer = csv.writer(court_file)
        player_writer = csv.writer(player_file)

        write_csv_header(court_writer)
        write_csv_header(player_writer)

        for packet in read_video(VIDEO_PATH):
            frame_id = packet.frame_id
            timestamp = packet.timestamp
            frame = packet.frame_bgr
            fps = packet.fps

            original_h, original_w = frame.shape[:2]

            frame_limit = max_frames if max_frames is not None else MAX_FRAMES
            if frame_limit is not None and frame_id >= frame_limit:
                break

            if frame_id % COURT_DETECTION_INTERVAL_FRAMES == 0 or last_court_points is None:
                court_tensor = preprocess_court_frame(frame)

                with torch.inference_mode():
                    court_out = court_model(court_tensor)

                court_points = extract_court_keypoints(
                    court_out=court_out,
                    original_w=original_w,
                    original_h=original_h,
                )

                last_court_points = court_points
            else:
                court_points = last_court_points

            for point in court_points:
                court_writer.writerow([
                    frame_id,
                    timestamp,
                    "court_model",
                    "court_point",
                    point["id"],
                    point["x"],
                    point["y"],
                    "",
                    "",
                    point["confidence"],
                    "",
                ])

            player_results = player_model.track(
                source=frame,
                tracker=PLAYER_TRACKER,
                conf=PLAYER_CONF,
                iou=PLAYER_IOU,
                imgsz=PLAYER_IMG_SIZE,
                persist=True,
                verbose=False,
            )

            candidates = get_candidates(
                results=player_results,
                frame_width=original_w,
                frame_height=original_h,
            )

            player_filter_H = build_player_filter_homography(court_points)
            candidates = attach_court_positions_to_candidates(candidates, player_filter_H)

            selected_players = select_two_players(
                candidates=candidates,
                frame_width=original_w,
                frame_height=original_h,
            )

            for object_id, player in selected_players.items():
                x1, y1, x2, y2 = player["bbox"]
                foot_x, foot_y = player["foot"]

                player_writer.writerow([
                    frame_id,
                    timestamp,
                    "yolo",
                    "player",
                    object_id,
                    x1,
                    y1,
                    x2 - x1,
                    y2 - y1,
                    player["confidence"],
                    player["track_id"],
                ])

                player_writer.writerow([
                    frame_id,
                    timestamp,
                    "yolo",
                    "player_foot",
                    f"{object_id}_foot",
                    foot_x,
                    foot_y,
                    "",
                    "",
                    player["confidence"],
                    player["track_id"],
                ])

            ball_x, ball_y = infer_ball_from_packet(
                packet=packet,
                model=ball_model,
                original_w=original_w,
                original_h=original_h,
                tracker=ball_motion_tracker,
            )

            ball_track.append((ball_x, ball_y))

            if frame_id % 100 == 0:
                print(f"Processed frame {frame_id}")

    if fps is None:
        raise RuntimeError("No frames were read from video.")

    raw_count = sum(1 for point in ball_track if is_valid_point(point))

    ball_track = remove_motion_outliers(ball_track)
    ball_track = remove_short_runs(ball_track)

    cleaned_count = sum(1 for point in ball_track if is_valid_point(point))

    final_count = sum(1 for point in ball_track if is_valid_point(point))

    save_ball_csv(
        ball_track=ball_track,
        path_output_csv=ball_csv,
        fps=fps,
    )

    print("Prediction done.")
    print(f"Ball detections: raw={raw_count}, cleaned={cleaned_count}, final={final_count}")
    print(f"Saved ball CSV: {ball_csv}")
    print(f"Saved court CSV: {court_csv}")
    print(f"Saved player CSV: {player_csv}")