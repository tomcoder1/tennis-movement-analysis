from dataclasses import dataclass
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier

WINDOW_RADIUS = 5
BOUNCE_THRESHOLD = 0.58
MODEL_PATH = Path(__file__).with_name("model") / "bounce_classifier.cbm"

@dataclass(frozen=True)
class Bounce:
    frame: int
    timestamp: float
    confidence: float

def _fill_short_gaps(values, limit=2):
    values = np.asarray(values, dtype=float).copy()
    valid = np.flatnonzero(np.isfinite(values))
    for left, right in zip(valid, valid[1:]):
        gap = right - left - 1
        if 0 < gap <= limit:
            values[left:right + 1] = np.linspace(values[left], values[right], gap + 2)
    return values

def trajectory_features(x_values, y_values, centers=None):
    x_values = _fill_short_gaps(x_values)
    y_values = _fill_short_gaps(y_values)
    centers = range(len(x_values)) if centers is None else centers
    padded_x = np.pad(x_values, WINDOW_RADIUS, constant_values=np.nan)
    padded_y = np.pad(y_values, WINDOW_RADIUS, constant_values=np.nan)
    rows = []
    for center in centers:
        x = padded_x[center:center + 2 * WINDOW_RADIUS + 1]
        y = padded_y[center:center + 2 * WINDOW_RADIUS + 1]
        speed = np.hypot(np.diff(x), np.diff(y))
        finite_speed = speed[np.isfinite(speed)]
        scale = np.median(finite_speed) if len(finite_speed) else 1.0
        if not np.isfinite(scale) or scale < 1.0:
            scale = 1.0
        vx, vy = np.diff(x) / scale, np.diff(y) / scale
        rows.append(np.concatenate((
            (x - x[WINDOW_RADIUS]) / scale, (y - y[WINDOW_RADIUS]) / scale,
            vx, vy, np.diff(vx), np.diff(vy), speed / scale,
        )))
    return np.asarray(rows, dtype=float)

class BounceDetector:
    def __init__(self, model_path=MODEL_PATH, threshold=BOUNCE_THRESHOLD):
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Bounce model not found: {self.model_path}")
        self.model = CatBoostClassifier()
        self.model.load_model(str(self.model_path))
        self.threshold = threshold

    def predict(self, image_ball):
        if not image_ball:
            return []
        first, last = min(image_ball), max(image_ball)
        frames = np.arange(first, last + 1)
        x, y = np.full(len(frames), np.nan), np.full(len(frames), np.nan)
        for frame, ball in image_ball.items():
            x[frame - first], y[frame - first] = ball.x, ball.y
        features = trajectory_features(x, y)
        valid = np.isfinite(features).sum(axis=1) >= 35
        probabilities = np.zeros(len(frames))
        if valid.any():
            probabilities[valid] = self.model.predict_proba(features[valid])[:, 1]
        episodes = []
        for index in np.flatnonzero(probabilities >= self.threshold):
            if not episodes or index - episodes[-1][-1] > 2:
                episodes.append([index])
            else:
                episodes[-1].append(index)
        result = []
        for episode in episodes:
            index = max(episode, key=lambda item: probabilities[item])
            frame = int(frames[index])
            nearest = min(image_ball.values(), key=lambda ball: abs(ball.frame - frame))
            result.append(Bounce(frame, nearest.timestamp, float(probabilities[index])))
        return result