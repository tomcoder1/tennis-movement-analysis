"""Detect coarse ball events from the CSV files produced by the pipeline.

Timing decisions are deliberately made in image space.  Homography values are
only attached after an event has been found, and are used to decide whether a
bounce was outside the court.
"""

import csv
import math
import os
from collections import defaultdict


BALL_CSV = "outputs/ball.csv"
BALL_HOMOGRAPHY_CSV = "outputs/ball_homography.csv"
PLAYER_CSV = "outputs/player.csv"
PLAYER_HOMOGRAPHY_CSV = "outputs/player_homography.csv"
OUTPUT_CSV = "outputs/ball_events.csv"

OUTPUT_FIELDS = [
    "frame_id", "timestamp", "event_type", "player_id", "img_x", "img_y",
    "court_x", "court_y", "confidence", "reason",
]

# These are intentionally collected here: footage with a different camera can
# normally be tuned without changing the detector below.
MAX_FILL_GAP = 3
NEAREST_FRAME_WINDOW = 3
VELOCITY_WINDOW = 2
HIT_COOLDOWN = 8
SAME_PLAYER_HIT_COOLDOWN = 24
BOUNCE_COOLDOWN = 10
TERMINAL_GAP = 15
COURT_MARGIN = 0.02


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _read_rows(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find CSV: {path}")
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _nearest(mapping, frame_id, window=NEAREST_FRAME_WINDOW):
    for distance in range(window + 1):
        offsets = (0,) if distance == 0 else (-distance, distance)
        for offset in offsets:
            if frame_id + offset in mapping:
                return mapping[frame_id + offset]
    return None


def _load_balls():
    balls = {}
    rows = _read_rows(BALL_CSV)
    last_csv_frame = max((int(row["frame_id"]) for row in rows), default=0)
    for row in rows:
        x, y = _number(row.get("x")), _number(row.get("y"))
        if x is not None and y is not None:
            frame = int(row["frame_id"])
            balls[frame] = {
                "frame_id": frame,
                "timestamp": _number(row.get("timestamp")) or 0.0,
                "x": x,
                "y": y,
                "confidence": _number(row.get("confidence")) or 0.0,
                "interpolated": False,
            }

    # Linear interpolation is only used across tiny detector dropouts.
    frames = sorted(balls)
    for left, right in zip(frames, frames[1:]):
        gap = right - left
        if 1 < gap <= MAX_FILL_GAP + 1:
            a, b = balls[left], balls[right]
            for frame in range(left + 1, right):
                ratio = (frame - left) / gap
                balls[frame] = {
                    "frame_id": frame,
                    "timestamp": a["timestamp"] + ratio * (b["timestamp"] - a["timestamp"]),
                    "x": a["x"] + ratio * (b["x"] - a["x"]),
                    "y": a["y"] + ratio * (b["y"] - a["y"]),
                    "confidence": min(a["confidence"], b["confidence"]) * 0.75,
                    "interpolated": True,
                }
    return balls, last_csv_frame


def _load_players():
    players = defaultdict(list)
    for row in _read_rows(PLAYER_CSV):
        if row.get("object_type") != "player":
            continue
        values = [_number(row.get(key)) for key in ("x", "y", "w", "h")]
        if any(value is None for value in values):
            continue
        x, y, w, h = values
        players[int(row["frame_id"])].append({
            "player_id": row.get("object_id", "").replace("_foot", ""),
            "x": x, "y": y, "w": w, "h": h,
            "confidence": _number(row.get("confidence")) or 0.0,
        })
    return players


def _load_homography(path, id_key=None):
    values = defaultdict(list) if id_key else {}
    if not os.path.exists(path):
        return values
    for row in _read_rows(path):
        if row.get("valid", "1") not in ("1", "1.0", "True", "true"):
            continue
        court_x, court_y = _number(row.get("court_x")), _number(row.get("court_y"))
        if court_x is None or court_y is None:
            continue
        item = {
            "court_x": court_x,
            "court_y": court_y,
            "confidence": _number(row.get("confidence")) or 0.0,
        }
        if id_key:
            item["player_id"] = row.get(id_key, "").replace("_foot", "")
            values[int(row["frame_id"])].append(item)
        else:
            values[int(row["frame_id"])] = item
    return values


def _velocity(balls, frame, start_offset, end_offset):
    a, b = balls.get(frame + start_offset), balls.get(frame + end_offset)
    if not a or not b:
        return None
    span = end_offset - start_offset
    return ((b["x"] - a["x"]) / span, (b["y"] - a["y"]) / span)


def _nearest_player(players, player_h, frame, ball):
    candidates = _nearest(players, frame, NEAREST_FRAME_WINDOW) or []
    best = None
    for player in candidates:
        # Distance to the box is more reliable than distance to its centre.
        dx = max(player["x"] - ball["x"], 0.0, ball["x"] - (player["x"] + player["w"]))
        dy = max(player["y"] - ball["y"], 0.0, ball["y"] - (player["y"] + player["h"]))
        distance = math.hypot(dx, dy)
        reach = max(45.0, 0.35 * math.hypot(player["w"], player["h"]))
        item = (distance / reach, distance, player)
        if best is None or item[0] < best[0]:
            best = item

    if best is None:
        return None, float("inf"), 0.0

    player = best[2]
    side_confidence = 1.0
    h_players = _nearest(player_h, frame, NEAREST_FRAME_WINDOW) or []
    matching = [p for p in h_players if p["player_id"] == player["player_id"]]
    if matching:
        # Side information is not used to time the event, only to strengthen
        # a stable player association.
        side_confidence = 0.8 + 0.2 * matching[0]["confidence"]
    return player["player_id"], best[0], player["confidence"] * side_confidence


def _event(frame, event_type, ball, ball_h, player_id="", confidence=0.5, reason=""):
    location = _nearest(ball_h, frame, NEAREST_FRAME_WINDOW) or {}
    return {
        "frame_id": frame,
        "timestamp": f'{ball["timestamp"]:.6f}',
        "event_type": event_type,
        "player_id": player_id,
        "img_x": f'{ball["x"]:.3f}',
        "img_y": f'{ball["y"]:.3f}',
        "court_x": "" if location.get("court_x") is None else f'{location["court_x"]:.6f}',
        "court_y": "" if location.get("court_y") is None else f'{location["court_y"]:.6f}',
        "confidence": f'{max(0.05, min(0.99, confidence)):.3f}',
        "reason": reason,
    }


def detect_ball_events():
    """Detect raw tennis-ball events and write ``outputs/ball_events.csv``."""
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    balls, last_csv_frame = _load_balls()
    players = _load_players()
    ball_h = _load_homography(BALL_HOMOGRAPHY_CSV)
    player_h = _load_homography(PLAYER_HOMOGRAPHY_CSV, "player_id")

    events = []
    last_hit = last_bounce = -10_000
    last_hitter = ""
    last_racket_hit_by_player = {}
    frames = sorted(balls)

    for frame in frames:
        ball = balls[frame]
        before = _velocity(balls, frame, -VELOCITY_WINDOW, 0)
        after = _velocity(balls, frame, 0, VELOCITY_WINDOW)
        if after is None:
            continue

        player_id, player_distance, player_conf = _nearest_player(players, player_h, frame, ball)
        speed_before = math.hypot(*before) if before else 0.0
        speed_after = math.hypot(*after)
        impulse = math.hypot(after[0] - before[0], after[1] - before[1]) if before else speed_after
        dot = before[0] * after[0] + before[1] * after[1] if before else 0.0
        direction_change = bool(before) and (
            dot < 0 or impulse > max(8.0, 0.45 * max(speed_before, speed_after))
        )

        # A toss begins close to a player and changes from nearly stationary or
        # downward motion into a clear upward image-space trajectory.
        is_toss = (
            player_id and player_distance <= 1.0 and after[1] < -5.0
            and abs(after[0]) < 10.0
            and (before is None or before[1] > -3.0)
            and frame - last_hit >= HIT_COOLDOWN
        )
        if is_toss:
            confidence = 0.55 + 0.20 * min(1.0, -after[1] / 20.0) + 0.15 * player_conf
            events.append(_event(
                frame, "serve_toss", ball, ball_h, player_id, confidence,
                "ball launched upward beside player",
            ))
            last_hit = frame  # suppress a duplicate hit at the launch instant
            last_hitter = player_id
            continue

        is_hit = (
            before is not None and player_id and player_distance <= 1.15 and direction_change
            and max(speed_before, speed_after) >= 4.0 and frame - last_hit >= HIT_COOLDOWN
            and frame - last_racket_hit_by_player.get(player_id, -10_000) >= SAME_PLAYER_HIT_COOLDOWN
        )
        if is_hit:
            confidence = 0.48 + 0.22 * min(1.0, impulse / 25.0) + 0.18 * player_conf
            events.append(_event(
                frame, "hit", ball, ball_h, player_id, confidence,
                "trajectory changed near player box",
            ))
            last_hit, last_hitter = frame, player_id
            last_racket_hit_by_player[player_id] = frame
            continue

        # In image coordinates a ground bounce normally changes vertical motion
        # from down (+y) to up (-y). Requiring distance from players avoids most
        # racket contacts being labelled as bounces.
        is_bounce = (
            before is not None and before[1] > 2.5 and after[1] < -2.5 and player_distance > 1.15
            and frame - last_bounce >= BOUNCE_COOLDOWN
        )
        if is_bounce:
            confidence = 0.52 + 0.25 * min(1.0, (before[1] - after[1]) / 25.0)
            events.append(_event(frame, "bounce", ball, ball_h, confidence=confidence,
                                 reason="vertical trajectory reversed at court level"))
            last_bounce = frame
            location = _nearest(ball_h, frame, NEAREST_FRAME_WINDOW)
            if location and not (
                -COURT_MARGIN <= location["court_x"] <= 1.0 + COURT_MARGIN
                and -COURT_MARGIN <= location["court_y"] <= 1.0 + COURT_MARGIN
            ):
                events.append(_event(frame, "out", ball, ball_h, confidence=confidence * 0.9,
                                     reason="bounce homography lies outside court bounds"))

    # Long observation gaps following a hit are weak but useful terminal evidence.
    # They are deliberately lower confidence than trajectory-based events.
    existing_terminal_frames = set()
    hits = [e for e in events if e["event_type"] == "hit"]
    gap_pairs = list(zip(frames, frames[1:]))
    if frames and last_csv_frame - frames[-1] > TERMINAL_GAP:
        gap_pairs.append((frames[-1], last_csv_frame))
    for left, right in gap_pairs:
        if right - left <= TERMINAL_GAP:
            continue
        previous_hits = [e for e in hits if 0 <= left - e["frame_id"] <= 90]
        if not previous_hits or left in existing_terminal_frames:
            continue
        hit = previous_hits[-1]
        victim = "player_2" if hit["player_id"] == "player_1" else "player_1"
        bounced = any(e["event_type"] == "bounce" and hit["frame_id"] < e["frame_id"] <= left for e in events)
        kind = "failed_return" if bounced else "miss"
        reason = "ball track ended after bounce without a return" if bounced else "ball track ended without an opponent contact"
        events.append(_event(left, kind, balls[left], ball_h, victim, 0.42, reason))
        existing_terminal_frames.add(left)

    event_order = {"serve_toss": 0, "hit": 1, "bounce": 2, "out": 3,
                   "miss": 4, "failed_return": 5}
    events.sort(key=lambda item: (item["frame_id"], event_order.get(item["event_type"], 9)))
    with open(OUTPUT_CSV, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(events)

    print(f"Saved {len(events)} ball events to: {OUTPUT_CSV}")
    return events


if __name__ == "__main__":
    detect_ball_events()
