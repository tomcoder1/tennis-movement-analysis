import math
from dataclasses import dataclass

EPISODE_SECONDS = 1.0
MIN_IMPULSE = 8.0
MAX_PROXIMITY = 1.15

@dataclass(frozen=True)
class ContactEpisode:
    frame: int
    timestamp: float
    player_id: str
    confidence: float
    impulse: float
    proximity: float
    speed_before: float
    speed_after: float
    start_frame: int = 0
    end_frame: int = 0
    raw_count: int = 1
    precheck_reason: str = ""

def _velocity(ball_by_frame, frame, first, second):
    a, b = ball_by_frame.get(frame + first), ball_by_frame.get(frame + second)
    if not a or not b:
        return None
    span = second - first
    return (b.x - a.x) / span, (b.y - a.y) / span

def _nearest_player(player_boxes, frame, ball):
    candidates = []
    for distance in range(4):
        candidates = player_boxes.get(frame - distance) or player_boxes.get(frame + distance) or []
        if candidates:
            break
    best = None
    for player in candidates:
        dx = max(player.x - ball.x, 0.0, ball.x - (player.x + player.w))
        dy = max(player.y - ball.y, 0.0, ball.y - (player.y + player.h))
        reach = max(45.0, 0.35 * math.hypot(player.w, player.h))
        proximity = math.hypot(dx, dy) / reach
        if best is None or proximity < best[0]:
            best = (proximity, player)
    return (best[1], best[0]) if best else (None, float("inf"))

def _candidate(frame, ball, tracks):
    before = _velocity(tracks.image_ball, frame, -2, 0)
    after = _velocity(tracks.image_ball, frame, 0, 2)
    if not before or not after:
        return None
    player, proximity = _nearest_player(tracks.player_boxes, frame, ball)
    if not player or proximity > 1.35:
        return None
    speed_before, speed_after = math.hypot(*before), math.hypot(*after)
    impulse = math.hypot(after[0] - before[0], after[1] - before[1])
    changed = before[0] * after[0] + before[1] * after[1] < 0
    if max(speed_before, speed_after) < 4.0:
        return None
    if not changed and impulse < 5.0:
        return None
    reason = ""
    if proximity > MAX_PROXIMITY:
        reason = "weak player proximity"
    elif impulse < MIN_IMPULSE:
        reason = "weak impulse"
    confidence = min(
        0.98,
        0.42 + 0.28 * min(1.0, impulse / 30.0)
        + 0.18 * player.confidence + 0.10 * max(0.0, 1.0 - proximity),
    )
    return ContactEpisode(
        frame, ball.timestamp, player.player_id, confidence, impulse, proximity,
        speed_before, speed_after, precheck_reason=reason,
    )

def _episode_choice(group):
    usable = [item for item in group if not item.precheck_reason]
    pool = usable or group

    best = max(pool, key=lambda item: (item.impulse * max(0.2, 1.25 - item.proximity), item.confidence))
    return ContactEpisode(
        best.frame, best.timestamp, best.player_id, best.confidence, best.impulse,
        best.proximity, best.speed_before, best.speed_after,
        min(item.frame for item in group), max(item.frame for item in group), len(group),
        best.precheck_reason,
    )

def detect_clip_boundaries(player_court, minimum_jump=0.25):
    by_player = {
        player_id: {point.frame: point for point in points}
        for player_id, points in player_court.items()
    }
    p1, p2 = by_player.get("player_1", {}), by_player.get("player_2", {})
    common = sorted(set(p1) & set(p2))
    boundaries = []
    for earlier, later in zip(common, common[1:]):
        if later - earlier > 2:
            continue
        jumps = [
            math.hypot(player[later].x - player[earlier].x, player[later].y - player[earlier].y)
            for player in (p1, p2)
        ]
        if min(jumps) >= minimum_jump:
            boundaries.append(later)
    return boundaries

def detect_contact_episodes(tracks, reset_frames=()):
    raw = []
    for frame in sorted(tracks.image_ball):
        candidate = _candidate(frame, tracks.image_ball[frame], tracks)
        if candidate:
            raw.append(candidate)

    by_player = {}
    for candidate in raw:
        by_player.setdefault(candidate.player_id, []).append(candidate)
    episodes = []
    for candidates in by_player.values():
        groups = []
        for candidate in candidates:
            crosses_reset = groups and any(
                groups[-1][-1].frame < reset <= candidate.frame for reset in reset_frames
            )
            if (groups and not crosses_reset
                    and candidate.timestamp - groups[-1][-1].timestamp <= EPISODE_SECONDS):
                groups[-1].append(candidate)
            else:
                groups.append([candidate])
        episodes.extend(_episode_choice(group) for group in groups)
    return sorted(episodes, key=lambda item: item.frame)