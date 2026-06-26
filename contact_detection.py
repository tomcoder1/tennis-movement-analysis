import math
from dataclasses import dataclass

EPISODE_SECONDS = 1.0
MIN_IMPULSE = 8.0
MAX_PROXIMITY = 1.15
RECOVERED_MIN_GAP_FRAMES = 3
RECOVERED_MAX_GAP_FRAMES = 14
RECOVERED_DUPLICATE_FRAMES = 12
RECOVERED_MAX_PROXIMITY = 1.05
RECOVERED_MIN_APPROACH_SPEED = 6.0
SOFT_VECTOR_INNER_FRAMES = 3
SOFT_VECTOR_OUTER_FRAMES = 10
SOFT_MAX_PROXIMITY = 1.25
SOFT_MIN_IMPULSE = 6.0
SOFT_MIN_ANGLE_CHANGE = 0.30
COURT_MAX_PROXIMITY = 0.22
COURT_MIN_SPEED = 0.015
COURT_MIN_DIRECTION_CHANGE = 0.20

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
    debug_reason: str = ""
    contact_source: str = "image"
    court_proximity: float = 0.0
    court_direction_change: float = 0.0

def _velocity(ball_by_frame, frame, first, second):
    a, b = ball_by_frame.get(frame + first), ball_by_frame.get(frame + second)
    if not a or not b:
        return None
    span = second - first
    return (b.x - a.x) / span, (b.y - a.y) / span

def _nearest_point(points, frame, window=3):
    nearby = [point for point in points if abs(point.frame - frame) <= window]
    return min(nearby, key=lambda point: abs(point.frame - frame)) if nearby else None

def _court_velocity(tracks, frame, first, second):
    a = tracks.ball_court_by_frame.get(frame + first)
    b = tracks.ball_court_by_frame.get(frame + second)
    if not a or not b:
        return None
    span = second - first
    return (b.x - a.x) / span, (b.y - a.y) / span

def _court_evidence(tracks, frame, player_id):
    ball = tracks.ball_court_by_frame.get(frame)
    player = _nearest_point(tracks.player_court.get(player_id, []), frame, window=4)
    before = _court_velocity(tracks, frame, -3, 0)
    after = _court_velocity(tracks, frame, 0, 3)
    if not ball or not player or not before or not after:
        return None

    proximity = math.hypot(ball.x - player.x, ball.y - player.y)
    speed_before, speed_after = math.hypot(*before), math.hypot(*after)
    if max(speed_before, speed_after) < COURT_MIN_SPEED:
        return None

    dot = before[0] * after[0] + before[1] * after[1]
    denom = max(1e-6, speed_before * speed_after)
    direction_change = 1.0 - max(-1.0, min(1.0, dot / denom))

    to_player = (player.x - ball.x, player.y - ball.y)
    from_player = (ball.x - player.x, ball.y - player.y)
    to_len = max(1e-6, math.hypot(*to_player))
    from_len = max(1e-6, math.hypot(*from_player))
    approach = (before[0] * to_player[0] + before[1] * to_player[1]) / (speed_before * to_len)
    away = (after[0] * from_player[0] + after[1] * from_player[1]) / (speed_after * from_len)

    near = proximity <= COURT_MAX_PROXIMITY
    changed = direction_change >= COURT_MIN_DIRECTION_CHANGE
    player_side_change = before[1] * after[1] < 0
    plausible = near and (changed or player_side_change) and (approach > -0.25 or away > -0.25)

    confidence = 0.0
    if plausible:
        confidence = min(
            0.82,
            0.52
            + 0.16 * max(0.0, 1.0 - proximity / COURT_MAX_PROXIMITY)
            + 0.10 * min(1.0, direction_change)
            + 0.04 * max(0.0, approach)
            + 0.04 * max(0.0, away),
        )

    return {
        "plausible": plausible,
        "confidence": confidence,
        "proximity": proximity,
        "direction_change": direction_change,
        "speed_before": speed_before,
        "speed_after": speed_after,
    }

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
        court = None
        player, proximity = _nearest_player(tracks.player_boxes, frame, ball)
        if player:
            court = _court_evidence(tracks, frame, player.player_id)
        if not court or not court["plausible"]:
            return None
    else:
        court = None
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
    if court is None:
        court = _court_evidence(tracks, frame, player.player_id)
    contact_source = "image"
    court_proximity = 0.0
    court_direction_change = 0.0
    if court:
        court_proximity = court["proximity"]
        court_direction_change = court["direction_change"]
        if court["plausible"]:
            confidence = min(0.98, max(confidence, court["confidence"]) + 0.04)
            contact_source = "hybrid"
        elif proximity > MAX_PROXIMITY or impulse < MIN_IMPULSE:
            reason = reason or "weak court confirmation"
    if getattr(ball, "state", "observed") != "observed":
        confidence = min(confidence, 0.72)
        reason = reason or "interpolated ball contact"
    return ContactEpisode(
        frame, ball.timestamp, player.player_id, confidence, impulse, proximity,
        speed_before, speed_after, precheck_reason=reason,
        contact_source=contact_source,
        court_proximity=court_proximity,
        court_direction_change=court_direction_change,
    )

def _episode_choice(group):
    usable = [item for item in group if not item.precheck_reason]
    pool = usable or group

    best = max(pool, key=lambda item: (item.impulse * max(0.2, 1.25 - item.proximity), item.confidence))
    return ContactEpisode(
        best.frame, best.timestamp, best.player_id, best.confidence, best.impulse,
        best.proximity, best.speed_before, best.speed_after,
        min(item.frame for item in group), max(item.frame for item in group), len(group),
        best.precheck_reason, best.debug_reason, best.contact_source,
        best.court_proximity, best.court_direction_change,
    )

def _player_center(player):
    return player.x + 0.5 * player.w, player.y + 0.5 * player.h

def _point_to_segment_distance(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-6:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))

def _approaching_player(ball_before, velocity, player):
    cx, cy = _player_center(player)
    to_player = (cx - ball_before.x, cy - ball_before.y)
    distance = math.hypot(*to_player)
    speed = math.hypot(*velocity)
    if distance <= 1e-6 or speed < RECOVERED_MIN_APPROACH_SPEED:
        return False, 0.0
    alignment = (velocity[0] * to_player[0] + velocity[1] * to_player[1]) / (speed * distance)
    return alignment >= 0.35, alignment

def _moving_away_from_player(ball_after, velocity, player):
    cx, cy = _player_center(player)
    from_player = (ball_after.x - cx, ball_after.y - cy)
    distance = math.hypot(*from_player)
    speed = math.hypot(*velocity)
    if distance <= 1e-6 or speed < 2.0:
        return False, 0.0
    alignment = (velocity[0] * from_player[0] + velocity[1] * from_player[1]) / (speed * distance)
    return alignment >= 0.15, alignment

def _velocity_between(a, b):
    span = b.frame - a.frame
    if span <= 0:
        return None
    return (b.x - a.x) / span, (b.y - a.y) / span

def _wide_velocity(ball_by_frame, frame, before=True):
    inner = SOFT_VECTOR_INNER_FRAMES
    outer = SOFT_VECTOR_OUTER_FRAMES
    offsets = ((-outer, -inner), (-outer + 2, -inner + 1), (-outer + 4, -inner + 2)) if before else (
        (inner, outer), (inner - 1, outer - 2), (inner - 2, outer - 4)
    )
    for first, second in offsets:
        velocity = _velocity(ball_by_frame, frame, first, second)
        if velocity:
            return velocity
    return None

def _recovered_candidate(before, after, tracks):
    gap = after.frame - before.frame
    if gap < RECOVERED_MIN_GAP_FRAMES or gap > RECOVERED_MAX_GAP_FRAMES:
        return None
    player, proximity = _nearest_player(tracks.player_boxes, before.frame + gap // 2, before)
    if not player or proximity > RECOVERED_MAX_PROXIMITY:
        return None
    path_distance = _point_to_segment_distance(
        player.x + 0.5 * player.w, player.y + 0.5 * player.h,
        before.x, before.y, after.x, after.y,
    )
    reach = max(45.0, 0.35 * math.hypot(player.w, player.h))
    if path_distance / reach > RECOVERED_MAX_PROXIMITY:
        return None

    prev = tracks.image_ball.get(before.frame - 2) or tracks.image_ball.get(before.frame - 1)
    nxt = tracks.image_ball.get(after.frame + 2) or tracks.image_ball.get(after.frame + 1)
    incoming = _velocity_between(prev, before) if prev else None
    outgoing = _velocity_between(after, nxt) if nxt else _velocity_between(before, after)
    if not incoming or not outgoing:
        return None

    approaching, approach_alignment = _approaching_player(before, incoming, player)
    away, away_alignment = _moving_away_from_player(after, outgoing, player)
    toward_net = (before.y < after.y <= 0.62) or (before.y > after.y >= 0.38)
    if not approaching or not (away or toward_net):
        return None

    confidence = min(
        0.75,
        0.55 + 0.10 * max(0.0, 1.0 - proximity)
        + 0.06 * max(0.0, approach_alignment)
        + 0.04 * max(0.0, away_alignment),
    )
    frame = before.frame + gap // 2
    timestamp = before.timestamp + 0.5 * (after.timestamp - before.timestamp)
    speed_before = math.hypot(*incoming)
    speed_after = math.hypot(*outgoing)
    impulse = math.hypot(outgoing[0] - incoming[0], outgoing[1] - incoming[1])
    return ContactEpisode(
        frame, timestamp, player.player_id, confidence, impulse, proximity,
        speed_before, speed_after,
        start_frame=before.frame, end_frame=after.frame, raw_count=1,
        debug_reason=(
            "recovered occlusion contact: "
            f"player={player.player_id[-1]}; frame={frame}; "
            "reason=ball disappeared near player and returned toward net"
        ),
        contact_source="gap_recovery",
    )

def _soft_visible_candidate(frame, ball, tracks):
    before = _wide_velocity(tracks.image_ball, frame, before=True)
    after = _wide_velocity(tracks.image_ball, frame, before=False)
    if not before or not after:
        return None
    player, proximity = _nearest_player(tracks.player_boxes, frame, ball)
    if not player or proximity > SOFT_MAX_PROXIMITY:
        return None
    speed_before, speed_after = math.hypot(*before), math.hypot(*after)
    if max(speed_before, speed_after) < 4.0:
        return None
    impulse = math.hypot(after[0] - before[0], after[1] - before[1])
    dot = before[0] * after[0] + before[1] * after[1]
    denom = max(1e-6, speed_before * speed_after)
    angle_change = 1.0 - max(-1.0, min(1.0, dot / denom))
    if impulse < SOFT_MIN_IMPULSE and angle_change < SOFT_MIN_ANGLE_CHANGE:
        return None
    confidence = min(
        0.75,
        0.56 + 0.08 * max(0.0, 1.0 - proximity)
        + 0.07 * min(1.0, impulse / 24.0)
        + 0.04 * min(1.0, angle_change),
    )
    return ContactEpisode(
        frame, ball.timestamp, player.player_id, confidence, impulse, proximity,
        speed_before, speed_after,
        debug_reason=(
            "recovered soft contact: "
            f"player={player.player_id[-1]}; frame={frame}; "
            "reason=wide-window trajectory change near player"
        ),
        contact_source="wide_window",
    )

def _recovered_contacts(tracks, normal_contacts):
    normal_by_frame = [(item.frame, item.player_id) for item in normal_contacts]
    frames = sorted(tracks.image_ball)
    recovered = []
    for earlier, later in zip(frames, frames[1:]):
        before, after = tracks.image_ball[earlier], tracks.image_ball[later]
        candidate = _recovered_candidate(before, after, tracks)
        if not candidate:
            continue
        if any(
            player_id == candidate.player_id
            and abs(frame - candidate.frame) <= RECOVERED_DUPLICATE_FRAMES
            for frame, player_id in normal_by_frame
        ):
            continue
        if any(
            item.player_id == candidate.player_id
            and abs(item.frame - candidate.frame) <= RECOVERED_DUPLICATE_FRAMES
            for item in recovered
        ):
            continue
        recovered.append(candidate)
    for frame in frames:
        candidate = _soft_visible_candidate(frame, tracks.image_ball[frame], tracks)
        if not candidate:
            continue
        if any(
            player_id == candidate.player_id
            and abs(existing_frame - candidate.frame) <= RECOVERED_DUPLICATE_FRAMES
            for existing_frame, player_id in normal_by_frame
        ):
            continue
        if any(
            item.player_id == candidate.player_id
            and abs(item.frame - candidate.frame) <= RECOVERED_DUPLICATE_FRAMES
            for item in recovered
        ):
            continue
        recovered.append(candidate)
    return recovered

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
    episodes.extend(_recovered_contacts(tracks, episodes))
    return sorted(episodes, key=lambda item: item.frame)
