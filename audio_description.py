"""Extract useful audio-description moments directly from tracking geometry."""

import csv
import math
from collections import defaultdict

from pipeline_utils import (
    DETECTION_COLUMNS, ensure_output_dirs, organized_path, parse_number,
    resolve_input_path, validate_csv,
)


MOMENT_FIELDS = [
    "moment_id", "start_timestamp", "end_timestamp", "moment_type",
    "player_id", "importance", "player_location", "player_movement",
    "ball_location", "ball_direction", "result", "confidence",
    "description_long", "description_short", "description_tiny",
]

NET_Y = 0.5
SINGLES_LEFT = 0.125
SINGLES_RIGHT = 0.875
VELOCITY_WINDOW = 2
CONTACT_COOLDOWN = 8
BOUNCE_COOLDOWN = 10
TRACK_END_GAP = 15


def player_name(player_id):
    clean = (player_id or "").replace("_foot", "")
    return "Player " + clean.split("_", 1)[1] if clean.startswith("player_") else (clean or "Player")


def service_side(x, y):
    if x is None or y is None:
        return ""
    deuce = (y >= NET_Y and x <= 0.5) or (y < NET_Y and x >= 0.5)
    return "deuce side" if deuce else "ad side"


def classify_player_location(x, y):
    if x is None or y is None:
        return ""
    if x < -0.08 or x > 1.08:
        return "outside court"
    if y < 0.0 or y > 1.0:
        return "behind baseline"
    if abs(y - NET_Y) <= 0.12:
        return "near net"
    if x < 0.16 or x > 0.84:
        return "wide"
    if 0.42 <= x <= 0.58:
        return "center"
    if y < 0.18 or y > 0.82:
        return "near baseline"
    return service_side(x, y)


def classify_ball_location(x, y):
    if x is None or y is None:
        return ""
    if x < 0.0 or x > 1.0:
        return "out wide"
    if y < 0.0 or y > 1.0:
        return "out long"
    if abs(y - NET_Y) <= 0.06:
        return "net area"
    if x < SINGLES_LEFT or x > SINGLES_RIGHT:
        return "sideline"
    if y <= 0.09 or y >= 0.91:
        return "baseline"
    if abs(x - SINGLES_LEFT) <= 0.05 or abs(x - SINGLES_RIGHT) <= 0.05:
        return "sideline"
    if abs(x - 0.5) <= 0.1:
        return "center court"
    if abs(y - NET_Y) >= 0.32:
        return "deep court"
    if abs(y - NET_Y) <= 0.27:
        return "service box"
    return "short court"


def classify_serve_target(x, y, server_y):
    location = classify_ball_location(x, y)
    if location in {"out wide", "out long", "net area"}:
        return location
    if abs(x - 0.5) <= 0.11:
        return "near the T"
    if min(abs(x - SINGLES_LEFT), abs(x - SINGLES_RIGHT)) <= 0.14:
        return "wide service box"
    return "service box"


def classify_shot_direction(contact, target):
    if not contact or not target or None in (*contact, *target):
        return "unknown"
    cx, _ = contact
    tx, ty = target
    if tx < SINGLES_LEFT - 0.03 or tx > SINGLES_RIGHT + 0.03:
        return "wide"
    if abs(tx - 0.5) <= 0.10:
        return "through the middle"
    depth = abs(ty - NET_Y)
    if depth >= 0.40:
        return "deep"
    if depth <= 0.18:
        return "short"
    return "crosscourt" if (cx - 0.5) * (tx - 0.5) < 0 else "down the line"


def _nearest_track_position(track, frame, window=3):
    candidates = [item for item in track if abs(item["frame"] - frame) <= window]
    return min(candidates, key=lambda item: abs(item["frame"] - frame)) if candidates else None


def classify_movement(track, frame, window=8):
    current = _nearest_track_position(track, frame)
    previous = _nearest_track_position(track, frame - window)
    if not current or not previous:
        return "", 0.3
    dx, dy = current["x"] - previous["x"], current["y"] - previous["y"]
    distance = math.hypot(dx, dy)
    confidence = min(current["confidence"], previous["confidence"], 1.0)
    near_side = previous["y"] >= NET_Y
    toward_net = -dy if near_side else dy
    if distance < 0.025:
        return "", confidence * 0.5
    if abs(current["x"] - 0.5) + 0.03 < abs(previous["x"] - 0.5):
        return "recovers to center", confidence * 0.85
    if abs(dx) > 0.09 and (current["x"] < 0.17 or current["x"] > 0.83):
        return "stretches wide", confidence * 0.85
    if toward_net > 0.04:
        return "approaches the net", confidence * 0.85
    if toward_net < -0.04 and (current["y"] < 0.05 or current["y"] > 0.95):
        return "retreats behind the baseline", confidence * 0.8
    side = service_side(current["x"], current["y"])
    return f"moves toward the {side}", confidence * 0.7


def make_descriptions(moment_type, player_id="", location="", movement="", direction="", result=""):
    name = player_name(player_id)
    direction = direction or "unknown"
    if moment_type == "serve_start":
        long = f"{name} serves from the {location}." if location else f"{name} serves."
        return long, f"{name} serves.", "Serve."
    if moment_type == "serve_result":
        if result in {"out wide", "out long"}:
            word = "wide" if result == "out wide" else "long"
            return f"The serve is {word}.", f"{word.title()} serve.", f"{word.title()}."
        if location == "wide service box":
            return "The serve lands wide in the service box.", "Wide serve.", "Wide."
        if location == "near the T":
            return "The serve lands near the T.", "Serve near the T.", "T serve."
        return "The serve lands in the service box.", "Serve in.", "In."
    if moment_type in {"return", "rally_shot", "deep_ball", "short_ball"}:
        action = "returns" if moment_type == "return" else "hits"
        direction_words = direction if direction != "unknown" else "the ball"
        long = f"{name} {movement} and {action} {direction_words}." if movement else f"{name} {action} {direction_words}."
        short = f"{name} {direction_words}." if direction != "unknown" else f"{name} hits."
        tiny = direction_words.title() + "." if direction != "unknown" else "Hit."
        return long, short, tiny
    if moment_type in {"wide_stretch", "recovery", "net_approach", "player_movement"}:
        return f"{name} {movement}.", f"{name} moves.", ""
    if moment_type == "ball_out":
        return "The ball lands out.", "Ball out.", "Out."
    if moment_type == "miss":
        return f"{name} misses the ball.", f"{name} misses.", "Missed."
    if moment_type == "point_result":
        if player_id:
            return f"Point to {name}.", f"Point {name}.", f"Point {name}."
        return "Point result unknown.", "Point unknown.", "Point."
    return result or "", result or "", ""


def _read_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _load_ball_image(path):
    balls = {}
    last_frame = 0
    for row in _read_rows(path):
        frame = int(float(row["frame_id"]))
        last_frame = max(last_frame, frame)
        x, y = parse_number(row.get("x")), parse_number(row.get("y"))
        if x is not None and y is not None:
            balls[frame] = {
                "frame": frame, "timestamp": parse_number(row.get("timestamp")) or 0.0,
                "x": x, "y": y, "confidence": parse_number(row.get("confidence")) or 0.0,
            }
    return balls, last_frame


def _load_player_boxes(path):
    players = defaultdict(list)
    for row in _read_rows(path):
        if row.get("object_type") != "player":
            continue
        values = [parse_number(row.get(key)) for key in ("x", "y", "w", "h")]
        if any(value is None for value in values):
            continue
        x, y, w, h = values
        players[int(float(row["frame_id"]))].append({
            "player_id": row.get("object_id", "").replace("_foot", ""),
            "x": x, "y": y, "w": w, "h": h,
            "confidence": parse_number(row.get("confidence")) or 0.0,
        })
    return players


def _load_geometry(path, by_player=False):
    result = defaultdict(list) if by_player else {}
    for row in _read_rows(path):
        if str(row.get("valid", "1")).lower() not in {"1", "1.0", "true"}:
            continue
        x, y = parse_number(row.get("court_x")), parse_number(row.get("court_y"))
        if x is None or y is None:
            continue
        item = {
            "frame": int(float(row["frame_id"])), "x": x, "y": y,
            "confidence": parse_number(row.get("confidence")) or 0.0,
        }
        if by_player:
            player_id = row.get("player_id", "").replace("_foot", "")
            result[player_id].append(item)
        else:
            result[item["frame"]] = item
    return result


def _nearest_frame(mapping, frame, window=3):
    for distance in range(window + 1):
        for offset in ((0,) if distance == 0 else (-distance, distance)):
            if frame + offset in mapping:
                return mapping[frame + offset]
    return None


def _velocity(balls, frame, first, second):
    a, b = balls.get(frame + first), balls.get(frame + second)
    if not a or not b:
        return None
    span = second - first
    return (b["x"] - a["x"]) / span, (b["y"] - a["y"]) / span


def _nearest_player(players, frame, ball):
    candidates = []
    for distance in range(4):
        candidates = players.get(frame - distance) or players.get(frame + distance) or []
        if candidates:
            break
    best = None
    for player in candidates:
        dx = max(player["x"] - ball["x"], 0.0, ball["x"] - (player["x"] + player["w"]))
        dy = max(player["y"] - ball["y"], 0.0, ball["y"] - (player["y"] + player["h"]))
        distance = math.hypot(dx, dy)
        reach = max(45.0, 0.35 * math.hypot(player["w"], player["h"]))
        score = distance / reach
        if best is None or score < best[0]:
            best = (score, player)
    return (best[1], best[0]) if best else (None, float("inf"))


def _opponent(player_id):
    return {"player_1": "player_2", "player_2": "player_1"}.get(player_id, "")


def _moment(timestamp, moment_type, player_id="", importance=2, player_location="",
            movement="", ball_location="", direction="", result="", confidence=0.5):
    long, short, tiny = make_descriptions(
        moment_type, player_id, player_location or ball_location, movement, direction, result,
    )
    return {
        "moment_id": "", "start_timestamp": timestamp,
        "end_timestamp": timestamp + 0.2, "moment_type": moment_type,
        "player_id": player_id, "importance": importance,
        "player_location": player_location, "player_movement": movement,
        "ball_location": ball_location, "ball_direction": direction,
        "result": result, "confidence": max(0.05, min(0.95, confidence)),
        "description_long": long, "description_short": short,
        "description_tiny": tiny,
    }


def extract_audio_description(output_dir="outputs"):
    ensure_output_dirs(output_dir)
    ball_csv = resolve_input_path(output_dir, "ball")
    player_csv = resolve_input_path(output_dir, "player")
    court_csv = resolve_input_path(output_dir, "court")
    ball_h_csv = resolve_input_path(output_dir, "ball_homography")
    player_h_csv = resolve_input_path(output_dir, "player_homography")
    for path in (ball_csv, player_csv, court_csv):
        validate_csv(path, DETECTION_COLUMNS, "predictor")
    validate_csv(ball_h_csv, {"frame_id", "court_x", "court_y", "valid", "confidence"}, "homography")
    validate_csv(player_h_csv, {"frame_id", "player_id", "court_x", "court_y", "valid", "confidence"}, "homography")

    balls, last_frame = _load_ball_image(ball_csv)
    player_boxes = _load_player_boxes(player_csv)
    ball_court = _load_geometry(ball_h_csv)
    player_tracks = _load_geometry(player_h_csv, by_player=True)
    frames = sorted(balls)
    contacts, bounces, tosses = [], [], []
    last_contact = last_bounce = last_toss = -10_000
    last_contact_by_player = {}

    for frame in frames:
        ball = balls[frame]
        before = _velocity(balls, frame, -VELOCITY_WINDOW, 0)
        after = _velocity(balls, frame, 0, VELOCITY_WINDOW)
        if not after:
            continue
        player, distance = _nearest_player(player_boxes, frame, ball)
        player_id = player["player_id"] if player else ""
        player_confidence = player["confidence"] if player else 0.0
        if (player_id and distance <= 1.0 and after[1] < -5.0 and abs(after[0]) < 10.0
                and (before is None or before[1] > -3.0) and frame - last_toss >= 45):
            tosses.append({"frame": frame, "player_id": player_id})
            last_toss = frame
            continue
        if before:
            speed_before, speed_after = math.hypot(*before), math.hypot(*after)
            impulse = math.hypot(after[0] - before[0], after[1] - before[1])
            changed = before[0] * after[0] + before[1] * after[1] < 0 or impulse > max(8.0, 0.45 * max(speed_before, speed_after))
            if (player_id and distance <= 1.15 and changed and max(speed_before, speed_after) >= 4.0
                    and frame - last_contact >= CONTACT_COOLDOWN
                    and frame - last_contact_by_player.get(player_id, -10_000) >= 24):
                launched_recently = any(
                    toss["player_id"] == player_id and 0 < frame - toss["frame"] < 8
                    for toss in tosses[-2:]
                )
                if launched_recently:
                    continue
                contacts.append({
                    "frame": frame, "timestamp": ball["timestamp"], "player_id": player_id,
                    "before_y": before[1],
                    "confidence": 0.5 + 0.2 * min(1.0, impulse / 25.0) + 0.15 * player_confidence,
                })
                last_contact = frame
                last_contact_by_player[player_id] = frame
                continue
            if (before[1] > 2.5 and after[1] < -2.5 and distance > 1.15
                    and frame - last_bounce >= BOUNCE_COOLDOWN):
                court = _nearest_frame(ball_court, frame)
                if court:
                    bounces.append({
                        "frame": frame, "timestamp": ball["timestamp"], "court": (court["x"], court["y"]),
                        "confidence": min(0.85, 0.5 + 0.2 * min(1.0, (before[1] - after[1]) / 25.0)),
                    })
                    last_bounce = frame

    moments = []
    server = ""
    return_seen = False
    shot_records = []
    used_tosses = set()
    for index, contact in enumerate(contacts):
        frame, player_id = contact["frame"], contact["player_id"]
        recent_toss = next((
            t for t in reversed(tosses)
            if 8 <= frame - t["frame"] <= 120 and t["player_id"] == player_id
            and t["frame"] not in used_tosses and contact.get("before_y", 0.0) > 0.0
        ), None)
        if recent_toss:
            shot_kind, server, return_seen = "serve_start", player_id, False
            used_tosses.add(recent_toss["frame"])
        elif server and player_id != server and not return_seen:
            shot_kind, return_seen = "return", True
        else:
            shot_kind = "rally_shot"
        next_frame = contacts[index + 1]["frame"] if index + 1 < len(contacts) else last_frame + 1
        landing = next((bounce for bounce in bounces if frame < bounce["frame"] < next_frame), None)
        track = player_tracks.get(player_id, [])
        player_position = _nearest_track_position(track, frame)
        contact_point = (player_position["x"], player_position["y"]) if player_position else (None, None)
        player_location = classify_player_location(*contact_point)
        movement, movement_confidence = classify_movement(track, frame)
        target = landing["court"] if landing else None
        direction = classify_shot_direction(contact_point, target)
        confidence = contact["confidence"] * (landing["confidence"] if landing else 0.65)

        if shot_kind == "serve_start":
            serve_location = service_side(*contact_point)
            moments.append(_moment(contact["timestamp"], "serve_start", player_id, 1,
                                   serve_location, movement, confidence=confidence))
            if landing:
                target_location = classify_serve_target(*target, contact_point[1])
                result = target_location if target_location in {"out wide", "out long"} else "in"
                moments.append(_moment(landing["timestamp"], "serve_result", player_id, 2,
                                       serve_location, "", target_location, direction, result,
                                       confidence * landing["confidence"]))
        else:
            moment_type = "return" if shot_kind == "return" else (
                "deep_ball" if direction == "deep" else "short_ball" if direction == "short" else "rally_shot"
            )
            delivery_time = landing["timestamp"] if landing else contact["timestamp"]
            importance = 2 if direction != "unknown" or movement else 3
            moment_confidence = confidence if direction != "unknown" else confidence * max(0.6, movement_confidence)
            moments.append(_moment(delivery_time, moment_type, player_id, importance,
                                   player_location, movement,
                                   classify_ball_location(*target) if target else "",
                                   direction, confidence=moment_confidence))
        shot_records.append({**contact, "landing": landing, "kind": shot_kind})

    terminal_shots = set()
    gaps = list(zip(frames, frames[1:]))
    if frames and last_frame - frames[-1] > TRACK_END_GAP:
        gaps.append((frames[-1], last_frame))
    for left, right in gaps:
        if right - left <= TRACK_END_GAP:
            continue
        shot = next((item for item in reversed(shot_records) if 0 <= left - item["frame"] <= 90), None)
        if not shot or shot["frame"] in terminal_shots:
            continue
        victim, winner = _opponent(shot["player_id"]), shot["player_id"]
        timestamp = balls[left]["timestamp"]
        moments.append(_moment(timestamp, "miss", victim, 1, confidence=0.38))
        moments.append(_moment(timestamp, "point_result", winner, 1, result="likely", confidence=0.34))
        terminal_shots.add(shot["frame"])

    for bounce in bounces:
        location = classify_ball_location(*bounce["court"])
        if location not in {"out wide", "out long"}:
            continue
        preceding = next((shot for shot in reversed(shot_records) if shot["frame"] < bounce["frame"]), None)
        if not preceding:
            continue
        moments.append(_moment(bounce["timestamp"], "ball_out", preceding["player_id"], 1,
                               ball_location=location, result=location, confidence=bounce["confidence"]))
        winner = _opponent(preceding["player_id"])
        moments.append(_moment(bounce["timestamp"], "point_result", winner, 1,
                               result="likely", confidence=bounce["confidence"] * 0.75))

    # Remove exact duplicate physical descriptions while retaining delayed direction.
    moments.sort(key=lambda item: (item["start_timestamp"], item["importance"]))
    deduplicated = []
    seen = set()
    for moment in moments:
        key = (round(moment["start_timestamp"], 2), moment["moment_type"], moment["player_id"])
        if key in seen or not moment["description_short"]:
            continue
        seen.add(key)
        deduplicated.append(moment)
    for index, moment in enumerate(deduplicated, 1):
        moment["moment_id"] = f"M{index:04d}"
        for key in ("start_timestamp", "end_timestamp", "confidence"):
            moment[key] = f'{moment[key]:.6f}'

    output_path = organized_path(output_dir, "moments")
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MOMENT_FIELDS)
        writer.writeheader()
        writer.writerows(deduplicated)
    print(f"Saved {len(deduplicated)} audio-description moments: {output_path}")
    return deduplicated


if __name__ == "__main__":
    extract_audio_description()
