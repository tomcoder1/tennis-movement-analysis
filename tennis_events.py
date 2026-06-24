"""Interpret raw ball events as simple tennis actions and point outcomes."""

import csv
import math
import os
from collections import defaultdict


BALL_EVENTS_CSV = "outputs/ball_events.csv"
PLAYER_HOMOGRAPHY_CSV = "outputs/player_homography.csv"
OUTPUT_CSV = "outputs/tennis_events.csv"
OUTPUT_FIELDS = [
    "frame_id", "timestamp", "event_type", "player_id", "court_x", "court_y",
    "target_court_x", "target_court_y", "result", "confidence", "description",
]
PLAYER_WINDOW = 4


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _load(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find CSV: {path}")
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _players():
    by_frame = defaultdict(list)
    if not os.path.exists(PLAYER_HOMOGRAPHY_CSV):
        return by_frame
    for row in _load(PLAYER_HOMOGRAPHY_CSV):
        x, y = _number(row.get("court_x")), _number(row.get("court_y"))
        if row.get("valid", "1") not in ("1", "1.0") or x is None or y is None:
            continue
        by_frame[int(row["frame_id"])].append({
            "player_id": row.get("player_id", "").replace("_foot", ""),
            "court_x": x, "court_y": y,
        })
    return by_frame


def _player_location(players, frame, player_id):
    for distance in range(PLAYER_WINDOW + 1):
        offsets = (0,) if distance == 0 else (-distance, distance)
        for offset in offsets:
            for player in players.get(frame + offset, []):
                if player["player_id"] == player_id:
                    return player["court_x"], player["court_y"]
    return None, None


def _opponent(player_id):
    return {"player_1": "player_2", "player_2": "player_1"}.get(player_id, "")


def _make(raw, kind, player_id="", court=None, target=None, result="", confidence=None, description=""):
    court = court or (_number(raw.get("court_x")), _number(raw.get("court_y")))
    target = target or (None, None)
    conf = _number(raw.get("confidence")) if confidence is None else confidence
    return {
        "frame_id": int(raw["frame_id"]),
        "timestamp": raw.get("timestamp", ""),
        "event_type": kind,
        "player_id": player_id,
        "court_x": "" if court[0] is None else f"{court[0]:.6f}",
        "court_y": "" if court[1] is None else f"{court[1]:.6f}",
        "target_court_x": "" if target[0] is None else f"{target[0]:.6f}",
        "target_court_y": "" if target[1] is None else f"{target[1]:.6f}",
        "result": result,
        "confidence": f"{max(0.05, min(0.99, conf or 0.4)):.3f}",
        "description": description,
    }


def interpret_tennis_events():
    """Write a readable, rule-based interpretation of raw ball events."""
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    raw_events = _load(BALL_EVENTS_CSV)
    raw_events.sort(key=lambda row: int(row["frame_id"]))
    players = _players()
    interpreted = []
    server = last_hitter = ""
    waiting_for_serve = False
    rally_hits = 0

    def next_bounce(index):
        for later in raw_events[index + 1:]:
            if later["event_type"] in ("bounce", "out"):
                return _number(later.get("court_x")), _number(later.get("court_y"))
            if later["event_type"] in ("hit", "serve_toss", "miss", "failed_return"):
                break
        return None, None

    for index, raw in enumerate(raw_events):
        kind = raw["event_type"]
        player_id = raw.get("player_id", "")

        if kind == "serve_toss":
            server, last_hitter, waiting_for_serve, rally_hits = player_id, "", True, 0
            location = _player_location(players, int(raw["frame_id"]), player_id)
            interpreted.append(_make(raw, "serve_toss", player_id, court=location,
                                     description=f"{player_id} begins a serve toss"))
        elif kind == "hit":
            location = _player_location(players, int(raw["frame_id"]), player_id)
            target = next_bounce(index)
            if waiting_for_serve and player_id == server:
                event_type, description = "serve", f"{player_id} serves"
                waiting_for_serve, rally_hits = False, 1
            elif server and rally_hits == 1 and player_id != server:
                event_type, description = "return", f"{player_id} returns {server}'s serve"
                rally_hits += 1
            else:
                event_type, description = "rally_hit", f"{player_id} hits during the rally"
                rally_hits += 1
            interpreted.append(_make(raw, event_type, player_id, court=location, target=target,
                                     description=description))
            last_hitter = player_id
        elif kind == "bounce":
            x, y = _number(raw.get("court_x")), _number(raw.get("court_y"))
            inside = x is not None and y is not None and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
            event_type, result = ("bounce_in", "in") if inside else ("bounce_out", "out")
            interpreted.append(_make(raw, event_type, result=result,
                                     description=f"Ball bounces {result}"))
        elif kind == "out":
            interpreted.append(_make(raw, "out", last_hitter, result="out",
                                     description=f"{last_hitter or 'Ball'} sends the ball out"))
            winner = _opponent(last_hitter)
            if winner:
                interpreted.append(_make(raw, "point_won", winner, result="point_won",
                                         confidence=(_number(raw.get("confidence")) or 0.5) * 0.9,
                                         description=f"{winner} likely wins the point (ball out)"))
            server = last_hitter = ""
            waiting_for_serve, rally_hits = False, 0
        elif kind in ("miss", "failed_return"):
            interpreted.append(_make(raw, kind, player_id, result=kind,
                                     description=f"{player_id} likely {'misses the ball' if kind == 'miss' else 'fails to return after the bounce'}"))
            winner = _opponent(player_id)
            if winner:
                interpreted.append(_make(raw, "point_won", winner, result="point_won",
                                         confidence=(_number(raw.get("confidence")) or 0.4) * 0.85,
                                         description=f"{winner} likely wins the point"))
            server = last_hitter = ""
            waiting_for_serve, rally_hits = False, 0

    with open(OUTPUT_CSV, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(interpreted)
    print(f"Saved {len(interpreted)} interpreted events to: {OUTPUT_CSV}")
    return interpreted


if __name__ == "__main__":
    interpret_tennis_events()
