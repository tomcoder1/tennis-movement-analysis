import math
from collections import Counter
from dataclasses import dataclass
from statistics import median

MIN_CONTACT_SECONDS = 0.18
DUPLICATE_SECONDS = 1.0
OUT_MARGIN = 0.03
NET_ZONE = 0.09
DEAD_GAP_FRAMES = 12
NET_MISS_LOOKAHEAD_FRAMES = 60
NET_CROSSING_CONFIRM_FRAMES = 3
NET_CROSSING_MAX_GAP_FRAMES = 4
NET_DIE_WINDOW_FRAMES = 30
NET_STOP_IMAGE_RADIUS = 35.0
NET_STOP_STEP_PIXELS = 8.0
NET_PROGRESS_EPSILON = 0.04
PLAYER_SIDE_WINDOW_FRAMES = 8
RECEIVER_SIDE_MARGIN = 0.08
MISSED_RECEIVER_MAX_PROXIMITY = 3.60
EXPECTED_RECOVERY_MAX_PROXIMITY = 1.35
EXPECTED_RECOVERY_MIN_CONFIDENCE = 0.60

@dataclass(frozen=True)
class Event:
    frame: int
    timestamp: float
    event_type: str
    player: str
    confidence: float
    description: str
    source_reason: str

@dataclass(frozen=True)
class AcceptedContact:
    episode: object
    event_type: str
    expected_hitter_after: str
    reason: str

    def __getattr__(self, name):
        return getattr(self.episode, name)

@dataclass(frozen=True)
class RejectedContact:
    episode: object
    reason: str

@dataclass(frozen=True)
class SyntheticContactEpisode:
    frame: int
    timestamp: float
    player_id: str
    confidence: float
    impulse: float = 0.0
    proximity: float = 0.0
    speed_before: float = 0.0
    speed_after: float = 0.0
    start_frame: int = 0
    end_frame: int = 0
    raw_count: int = 1
    precheck_reason: str = ""
    debug_reason: str = ""

@dataclass(frozen=True)
class PointSegment:
    index: int
    start_frame: int
    end_frame: int
    contacts: tuple

@dataclass(frozen=True)
class Terminal:
    frame: int
    timestamp: float
    loser: str
    winner: str
    confidence: float
    reason: str
    miss_type: str

@dataclass(frozen=True)
class BuildResult:
    events: tuple
    accepted: tuple
    rejected: tuple
    missing_terminals: int
    miss_type_counts: dict

def player_number(player_id):
    return "1" if player_id == "player_1" else "2" if player_id == "player_2" else ""

def other_player(player_id):
    return "player_2" if player_id == "player_1" else "player_1"

def opponent(player):
    return "2" if player == "1" else "1" if player == "2" else ""

def lateral_area(x):
    if x < 0.44:
        return "left"
    if x > 0.56:
        return "right"
    return "middle"

def _nearest_point(points, frame, window=3):
    nearby = [point for point in points if abs(point.frame - frame) <= window]
    return min(nearby, key=lambda point: abs(point.frame - frame)) if nearby else None

def _ball_left_and_returned(tracks, previous, current):
    from_bottom = _player_from_bottom(tracks, previous.player_id, previous.frame)
    if from_bottom is not None:
        points = [
            point for point in tracks.ball_court
            if previous.frame + 4 <= point.frame <= current.frame - 4
            and point.confidence >= 0.45
        ]
        current_point = _nearest_point(tracks.ball_court, current.frame, window=4)
        if points and current_point:
            opponent_side_seen = any(
                point.y <= 0.5 - RECEIVER_SIDE_MARGIN
                if from_bottom else point.y >= 0.5 + RECEIVER_SIDE_MARGIN
                for point in points
            )
            returned_to_hitter_side = (
                current_point.y >= 0.5 + RECEIVER_SIDE_MARGIN
                if from_bottom else current_point.y <= 0.5 - RECEIVER_SIDE_MARGIN
            )
            return opponent_side_seen, returned_to_hitter_side

    origin = tracks.image_ball.get(previous.frame)
    destination = tracks.image_ball.get(current.frame)
    if not origin or not destination or current.frame - previous.frame < 12:
        return False, False
    middle = [
        ball for frame, ball in tracks.image_ball.items()
        if previous.frame + 4 <= frame <= current.frame - 4
    ]
    if not middle:
        return False, False
    farthest = max(math.hypot(ball.x - origin.x, ball.y - origin.y) for ball in middle)
    left = farthest >= 160.0
    returned = math.hypot(destination.x - origin.x, destination.y - origin.y) <= max(220.0, 0.65 * farthest)
    return left, returned

def _point_on_player_side(point, player_from_bottom, margin=0.04):
    if player_from_bottom is None:
        return False
    return point.y >= 0.5 + margin if player_from_bottom else point.y <= 0.5 - margin

def _expected_hitter_recovery_allowed(tracks, previous, episode, expected_hitter):
    if episode.player_id != expected_hitter:
        return False, ""
    if not episode.precheck_reason:
        return False, ""
    if episode.confidence < EXPECTED_RECOVERY_MIN_CONFIDENCE:
        return False, ""
    if episode.proximity > EXPECTED_RECOVERY_MAX_PROXIMITY:
        return False, ""
    if episode.frame - previous.frame < 12:
        return False, ""
    player_from_bottom = _player_from_bottom(tracks, episode.player_id, episode.frame)
    nearby_ball = _nearest_point(tracks.ball_court, episode.frame, window=8)
    if not nearby_ball or not _point_on_player_side(nearby_ball, player_from_bottom):
        return False, ""
    return True, (
        "accepted expected-hitter recovery: "
        f"player={player_number(episode.player_id)}; frame={episode.frame}; "
        "reason=filled missing rally contact before terminal"
    )

def _ball_timestamp(tracks, frame, fallback=0.0):
    ball = tracks.image_ball.get(frame)
    return ball.timestamp if ball else fallback

def _bounce_player_id(tracks, bounce):
    point = _nearest_point(tracks.ball_court, bounce.frame, window=2)
    if not point:
        return None
    return "player_1" if point.y >= 0.5 else "player_2"

def _has_contact_between(contacts, player_id, start_frame, end_frame):
    return any(
        contact.player_id == player_id and start_frame <= contact.frame <= end_frame
        for contact in contacts
    )

def _is_bounce_overlap_candidate(episode, bounces, window=6):
    return any(
        bounce.confidence >= 0.55 and abs(bounce.frame - episode.frame) <= window
        for bounce in bounces
    )

def _recover_bounce_contacts(tracks, candidates, bounces, segment_start, segment_end):
    if not candidates:
        return candidates
    recovered = []
    existing = [
        episode for episode in candidates
        if not _is_bounce_overlap_candidate(episode, bounces)
    ]
    usable_bounces = [
        bounce for bounce in bounces
        if segment_start < bounce.frame < segment_end and bounce.confidence >= 0.55
    ]
    for index, bounce in enumerate(usable_bounces[:-1]):
        next_bounce = usable_bounces[index + 1]
        if next_bounce.frame - bounce.frame < 14:
            continue
        player_id = _bounce_player_id(tracks, bounce)
        if not player_id:
            continue
        previous_contacts = [
            contact for contact in existing
            if contact.frame < bounce.frame
        ]
        if previous_contacts and previous_contacts[-1].player_id == player_id:
            player_id = other_player(previous_contacts[-1].player_id)
        if _has_contact_between(existing, player_id, bounce.frame - 4, next_bounce.frame - 4):
            continue
        if previous_contacts and previous_contacts[-1].player_id == player_id:
            continue
        frame = min(bounce.frame + 14, next_bounce.frame - 5)
        if any(abs(contact.frame - frame) <= 10 for contact in existing):
            continue
        recovered.append(SyntheticContactEpisode(
            frame=frame,
            timestamp=_ball_timestamp(tracks, frame, bounce.timestamp),
            player_id=player_id,
            confidence=min(0.74, max(0.62, bounce.confidence * 0.72)),
            start_frame=bounce.frame,
            end_frame=next_bounce.frame,
            debug_reason=(
                "recovered bounce rally contact: "
                f"player={player_number(player_id)}; frame={frame}; "
                f"bounce_frame={bounce.frame}; next_bounce_frame={next_bounce.frame}; "
                "reason=ball bounced on player side and rally continued"
            ),
        ))
        existing.append(recovered[-1])
        existing.sort(key=lambda episode: episode.frame)
    if not recovered:
        return candidates
    return sorted([*candidates, *recovered], key=lambda episode: episode.frame)

def _validate_segment(tracks, episodes):
    accepted, rejected = [], []
    expected_hitter = None
    for episode in episodes:
        if not accepted:
            if getattr(episode, "debug_reason", ""):
                rejected.append(RejectedContact(episode, "recovered contact before first serve"))
                continue
            if episode.precheck_reason:
                rejected.append(RejectedContact(episode, episode.precheck_reason))
                continue
            expected_hitter = other_player(episode.player_id)
            accepted.append(AcceptedContact(episode, "serve", expected_hitter, "first valid contact in clip"))
            continue
        previous = accepted[-1]
        elapsed = episode.timestamp - previous.timestamp
        same_player = episode.player_id == previous.player_id
        if elapsed < MIN_CONTACT_SECONDS:
            rejected.append(RejectedContact(episode, "too soon after previous contact"))
            continue
        if same_player and elapsed < DUPLICATE_SECONDS:
            reason = "serve follow-through duplicate" if previous.event_type == "serve" else "duplicate same-player contact"
            rejected.append(RejectedContact(episode, reason))
            continue
        if same_player:
            rejected.append(RejectedContact(episode, "missing opponent contact before same-player"))
            continue
        recovery_ok, recovery_reason = _expected_hitter_recovery_allowed(
            tracks, previous, episode, expected_hitter,
        )
        if episode.precheck_reason and not recovery_ok:
            rejected.append(RejectedContact(episode, episode.precheck_reason))
            continue
        if episode.player_id != expected_hitter:
            left, returned = _ball_left_and_returned(tracks, previous, episode)
            if not left:
                rejected.append(RejectedContact(episode, "ball did not leave and return"))
                continue
            if not returned:
                rejected.append(RejectedContact(episode, "wrong expected hitter"))
                continue
            reason = "same-player exception: ball clearly left and returned"
        else:
            reason = recovery_reason or "contact by expected receiver"
        expected_hitter = other_player(episode.player_id)
        accepted.append(AcceptedContact(episode, "hit", expected_hitter, reason))
    return accepted, rejected

def build_point_sequence(tracks, episodes, reset_frames, bounces=()):
    boundaries = [0, *sorted(reset_frames), tracks.last_frame + 1]
    segments, rejected = [], []
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:]), 1):
        candidates = [episode for episode in episodes if start <= episode.frame < end]
        long_rally = sum(1 for bounce in bounces if start < bounce.frame < end) >= 30
        if long_rally:
            candidates = _recover_bounce_contacts(tracks, candidates, bounces, start, end)
        overlaps = []
        for episode in candidates:
            if long_rally and "recovered soft contact" in getattr(episode, "debug_reason", ""):
                continue
            weak_or_recovered = episode.precheck_reason or getattr(episode, "debug_reason", "")
            bounce_window = 6 if weak_or_recovered else 2
            bounce_confidence = 0.60 if weak_or_recovered else 0.80
            if any(
                bounce.confidence >= bounce_confidence and abs(bounce.frame - episode.frame) <= bounce_window
                for bounce in bounces
            ):
                overlaps.append(episode)
        candidates = [episode for episode in candidates if episode not in overlaps]
        accepted, invalid = _validate_segment(tracks, candidates)
        segments.append(PointSegment(index, start, end, tuple(accepted)))
        rejected.extend(invalid)
        rejected.extend(RejectedContact(episode, "classified bounce overlap") for episode in overlaps)
    return segments, rejected

def _source_x(tracks, contact):
    source = _nearest_point(tracks.player_court.get(contact.player_id, []), contact.frame)
    return source.x if source and source.confidence >= 0.55 else 0.5

def _hitter_from_bottom(tracks, contact):
    return _player_from_bottom(tracks, contact.player_id, contact.frame)

def _player_from_bottom(tracks, player_id, frame):
    source = _nearest_point(
        tracks.player_court.get(player_id, []),
        frame,
        window=PLAYER_SIDE_WINDOW_FRAMES,
    )
    return source.y >= 0.5 if source and source.confidence >= 0.55 else None

def _serve_target_x(tracks, contact_frame, end_frame, bounces):
    for bounce in bounces:
        if contact_frame < bounce.frame <= end_frame:
            point = _nearest_point(tracks.ball_court, bounce.frame, window=2)
            if point and point.confidence >= 0.55:
                return point.x
    late_start = max(contact_frame + 3, end_frame - 18)
    late_points = [
        point for point in tracks.ball_court
        if late_start <= point.frame <= end_frame
        and point.confidence >= 0.55
    ]
    if len(late_points) >= 3:
        return median(point.x for point in late_points[-8:])
    points = [
        point for point in tracks.ball_court
        if contact_frame + 8 <= point.frame <= end_frame
        and point.confidence >= 0.55
    ]
    return median(point.x for point in points[-8:]) if len(points) >= 3 else None

def _median_point(points):
    if not points:
        return None
    return type(points[0])(
        int(round(median(point.frame for point in points))),
        median(point.timestamp for point in points),
        median(point.x for point in points),
        median(point.y for point in points),
        median(point.confidence for point in points),
    )

def _hit_target_point(tracks, contact, end_frame, bounces=(), next_contact=None, terminal=None):
    boundary = next_contact.frame if next_contact else end_frame + 1
    for bounce in bounces:
        if contact.frame < bounce.frame < boundary:
            point = _nearest_point(tracks.ball_court, bounce.frame, window=2)
            if point and point.confidence >= 0.55:
                return point, "bounce"

    if next_contact:
        points = [
            point for point in tracks.ball_court
            if next_contact.frame - 8 <= point.frame <= next_contact.frame - 3
            and point.confidence >= 0.55
        ]
        point = _median_point(points)
        if point:
            return point, "before_next_contact"

    if terminal:
        points = [
            point for point in tracks.ball_court
            if contact.frame < point.frame <= terminal.frame and point.confidence >= 0.55
        ]
        if points:
            return max(points, key=lambda point: point.frame), "terminal"

    points = [
        point for point in tracks.ball_court
        if contact.frame + 12 <= point.frame <= min(contact.frame + 24, end_frame)
        and point.confidence >= 0.55
    ]
    point = _median_point(points)
    return (point, "fallback") if point else (None, "unreliable")

def _hit_direction_detail(tracks, contact, end_frame, bounces=(), next_contact=None, terminal=None):
    start = _nearest_point(tracks.ball_court, contact.frame, window=3)
    target, target_source = _hit_target_point(
        tracks, contact, end_frame, bounces, next_contact, terminal,
    )
    if not start or start.confidence < 0.55 or not target:
        return {
            "target_source": target_source, "start_frame": None, "target_frame": None,
            "start_x": None, "target_x": None, "dx": None, "dy": None,
            "angle_deg": None, "final_direction": "",
        }
    target_x = target.x
    target_y = target.y
    dx = target_x - start.x
    dy = target_y - start.y
    angle = math.degrees(math.atan2(dx, abs(dy)))
    if angle <= -15.0:
        direction = "left"
    elif angle >= 15.0:
        direction = "right"
    elif abs(angle) <= 10:
        direction = "straight"
    else:
        direction = ""

    return {
        "target_source": target_source, "start_frame": start.frame,
        "target_frame": target.frame, "start_x": start.x, "target_x": target_x, "dx": dx, "dy": dy,
        "angle_deg": angle, "final_direction": direction,
    }

def _hit_direction(tracks, contact, end_frame, bounces=(), next_contact=None, terminal=None):
    return _hit_direction_detail(
        tracks, contact, end_frame, bounces, next_contact, terminal,
    )["final_direction"] or None

def _hit_direction_reason(tracks, contact, end_frame, bounces=(), next_contact=None, terminal=None):
    detail = _hit_direction_detail(tracks, contact, end_frame, bounces, next_contact, terminal)
    value = lambda key: "" if detail[key] is None else f"{detail[key]:.4f}"
    frame = lambda key: "" if detail[key] is None else str(detail[key])
    return (
        "hit_direction "
        f"target_source={detail['target_source']} "
        f"start_frame={frame('start_frame')} target_frame={frame('target_frame')} "
        f"start_x={value('start_x')} target_x={value('target_x')} "
        f"dx={value('dx')} dy={value('dy')} "
        f"angle_deg={value('angle_deg')} "
        f"final_direction={detail['final_direction'] or 'plain'}"
    )

def _contact_description(tracks, contact, end_frame, bounces=(), next_contact=None, terminal=None):
    player = player_number(contact.player_id)
    if contact.event_type == "serve":
        source = _source_x(tracks, contact)
        target = _serve_target_x(tracks, contact.frame, end_frame, bounces)
        target = source if target is None else target
        return f"{player} serve from {lateral_area(source)} to {lateral_area(target)}"
    direction = _hit_direction(tracks, contact, end_frame, bounces, next_contact, terminal)
    return f"{player} hit {direction}" if direction else f"{player} hit"

def _court_segment(tracks, start, boundary):
    return [point for point in tracks.ball_court if start < point.frame < boundary and point.confidence >= 0.50]

def _on_opponent_side(point, from_bottom):
    return point.y < 0.5 - NET_ZONE if from_bottom else point.y > 0.5 + NET_ZONE

def _confirmed_crossing(points, from_bottom):
    consecutive = 0
    previous_frame = None
    for point in points:
        if not _on_opponent_side(point, from_bottom):
            consecutive = 0
            previous_frame = None
            continue
        if previous_frame is not None and point.frame - previous_frame > NET_CROSSING_MAX_GAP_FRAMES:
            consecutive = 1
        else:
            consecutive += 1
        previous_frame = point.frame
        if consecutive >= NET_CROSSING_CONFIRM_FRAMES:
            return True, point.frame
    return False, None

def _image_segment(tracks, start, boundary):
    return [
        ball for frame, ball in sorted(tracks.image_ball.items())
        if start < frame < boundary
    ]

def _player_ball_proximity(tracks, player_id, frame, window=4):
    candidates = []
    for distance in range(window + 1):
        ball = tracks.image_ball.get(frame - distance) or tracks.image_ball.get(frame + distance)
        boxes = tracks.player_boxes.get(frame - distance) or tracks.player_boxes.get(frame + distance) or []
        box = next((item for item in boxes if item.player_id == player_id), None)
        if ball and box:
            candidates.append((ball, box))
    best = None
    for ball, box in candidates:
        dx = max(box.x - ball.x, 0.0, ball.x - (box.x + box.w))
        dy = max(box.y - ball.y, 0.0, ball.y - (box.y + box.h))
        reach = max(45.0, 0.35 * math.hypot(box.w, box.h))
        proximity = math.hypot(dx, dy) / reach
        best = proximity if best is None else min(best, proximity)
    return best

def _movement_steps(points):
    return [
        math.hypot(current.x - previous.x, current.y - previous.y)
        for previous, current in zip(points, points[1:])
    ]

def _timestamp_for_frame(tracks, frame, fallback):
    ball = tracks.image_ball.get(frame)
    if ball:
        return ball.timestamp
    point = _nearest_point(tracks.ball_court, frame, window=0)
    return point.timestamp if point else fallback

def _net_death_evidence(tracks, action, boundary, reached_net, from_bottom, bounces=()):
    window_end = min(boundary, reached_net.frame + NET_DIE_WINDOW_FRAMES + 1)
    near_net_points = [
        point for point in tracks.ball_court
        if reached_net.frame <= point.frame < window_end
        and point.confidence >= 0.50
        and abs(point.y - 0.5) <= NET_ZONE * 1.6
    ]

    near_bounces = [
        bounce for bounce in bounces
        if reached_net.frame <= bounce.frame < window_end
        and _nearest_point(near_net_points, bounce.frame, window=3) is not None
    ]
    if near_bounces:
        return near_bounces[0].frame, "near-net bounce"

    image_after_net = _image_segment(tracks, reached_net.frame - 1, boundary)
    if image_after_net:
        last_image = image_after_net[-1]
        if last_image.frame < window_end and boundary - last_image.frame >= DEAD_GAP_FRAMES:
            return last_image.frame, "ball disappeared after net"

    image_after_net = [ball for ball in image_after_net if ball.frame < window_end]
    if len(image_after_net) >= 4:
        origin = image_after_net[0]
        radius = max(math.hypot(ball.x - origin.x, ball.y - origin.y) for ball in image_after_net)
        steps = _movement_steps(image_after_net)
        median_step = median(steps) if steps else 0.0
        if radius <= NET_STOP_IMAGE_RADIUS and median_step <= NET_STOP_STEP_PIXELS:
            return image_after_net[-1].frame, (
                f"ball stalled near net radius={radius:.1f} median_step={median_step:.1f}"
            )

    later_near = near_net_points
    if len(later_near) >= 3:
        progress = (
            reached_net.y - median(point.y for point in later_near[-3:])
            if from_bottom else
            median(point.y for point in later_near[-3:]) - reached_net.y
        )
        if progress <= NET_PROGRESS_EPSILON:
            return later_near[-1].frame, f"ball lost progress near net progress={progress:.4f}"

    return None

def _net_death_terminal(tracks, player_id, boundary, reached_net, from_bottom, bounces, reason_prefix):
    evidence = _net_death_evidence(tracks, None, boundary, reached_net, from_bottom, bounces)
    if not evidence:
        return None
    death_frame, death_reason = evidence
    if not (reached_net.frame <= death_frame < boundary):
        return None
    loser = player_number(player_id)
    return Terminal(
        death_frame, _timestamp_for_frame(tracks, death_frame, reached_net.timestamp),
        loser, opponent(loser), 0.82,
        reason_prefix(death_frame, death_reason),
        "net",
    )

def _net_reach(points):
    reached = [point for point in points if abs(point.y - 0.5) <= NET_ZONE]
    if reached:
        return reached[0]
    closest = min(points, key=lambda point: abs(point.y - 0.5), default=None)
    if closest and abs(closest.y - 0.5) <= NET_ZONE * 1.6:
        return closest
    return None

def _net_fail(tracks, action, boundary, bounces=()):
    boundary = min(boundary, action.frame + NET_MISS_LOOKAHEAD_FRAMES + 1)
    points = _court_segment(tracks, action.frame + 1, boundary)
    if len(points) < 5:
        return None
    from_bottom = _hitter_from_bottom(tracks, action)
    if from_bottom is None:
        return None
    reached_net = _net_reach(points)
    if not reached_net:
        return None
    post_net_points = [point for point in points if point.frame >= reached_net.frame]
    crossed, _ = _confirmed_crossing(post_net_points, from_bottom)
    if crossed:
        return None
    return _net_death_terminal(
        tracks, action.player_id, boundary, reached_net, from_bottom, bounces,
        lambda death_frame, death_reason: (
            "net fail by hitter: "
            f"player={player_number(action.player_id)}; reached_net_frame={reached_net.frame}; "
            f"death_frame={death_frame}; confirmed_crossing=false; "
            f"reason={death_reason}"
        ),
    )

def _receiver_side_points(points, receiver_from_bottom):
    if receiver_from_bottom:
        return [point for point in points if point.y >= 0.5 + RECEIVER_SIDE_MARGIN]
    return [point for point in points if point.y <= 0.5 - RECEIVER_SIDE_MARGIN]

def _missed_receiver_net_fail(tracks, action, boundary, bounces):
    receiver = action.expected_hitter_after or other_player(action.player_id)
    receiver_from_bottom = _player_from_bottom(tracks, receiver, action.frame)
    if receiver_from_bottom is None:
        return None
    points = _court_segment(
        tracks, action.frame + 1,
        min(boundary, action.frame + NET_MISS_LOOKAHEAD_FRAMES + 1),
    )
    if len(points) < 6:
        return None
    receiver_side = _receiver_side_points(points, receiver_from_bottom)
    reached_net = _net_reach(points)
    if not reached_net:
        return None
    if not receiver_side:
        return None
    reached_receiver = receiver_side[0]
    receiver_bounces = []
    for bounce in bounces:
        if not reached_net.frame <= bounce.frame < boundary:
            continue
        proximity = _player_ball_proximity(tracks, receiver, bounce.frame)
        if proximity is not None and proximity <= MISSED_RECEIVER_MAX_PROXIMITY:
            receiver_bounces.append(bounce)
    if receiver_bounces:
        bounce = receiver_bounces[0]
        post_net_points = [
            point for point in points
            if reached_net.frame <= point.frame <= bounce.frame
        ]
        crossed, _ = _confirmed_crossing(post_net_points, receiver_from_bottom)
        if not crossed:
            loser = player_number(receiver)
            return Terminal(
                bounce.frame, bounce.timestamp, loser, opponent(loser), 0.82,
                (
                    "missed receiver net fail: "
                    f"previous_hitter={player_number(action.player_id)}; "
                    f"inferred_hitter={loser}; "
                    f"reached_receiver_side_frame={reached_receiver.frame}; "
                    f"reached_net_frame={reached_net.frame}; death_frame={bounce.frame}; "
                    "confirmed_crossing=false; reason=near-net bounce near inferred receiver"
                ),
                "net",
            )
    terminal = _net_death_terminal(
        tracks, receiver, boundary, reached_net, receiver_from_bottom, bounces,
        lambda death_frame, death_reason: (
            "missed receiver net fail: "
            f"previous_hitter={player_number(action.player_id)}; "
            f"inferred_hitter={player_number(receiver)}; "
            f"reached_receiver_side_frame={reached_receiver.frame}; "
            f"reached_net_frame={reached_net.frame}; death_frame={death_frame}; "
            "confirmed_crossing=false"
            f"; reason={death_reason}"
        ),
    )
    if not terminal:
        return None
    post_net_points = [
        point for point in points
        if reached_net.frame <= point.frame <= terminal.frame
    ]
    crossed, _ = _confirmed_crossing(post_net_points, receiver_from_bottom)
    if crossed:
        return None
    proximities = [
        value for value in (
            _player_ball_proximity(tracks, receiver, reached_net.frame),
            _player_ball_proximity(tracks, receiver, terminal.frame),
        )
        if value is not None
    ]
    if not proximities or min(proximities) > MISSED_RECEIVER_MAX_PROXIMITY:
        return None
    return terminal

def _near_net_terminal_fail(tracks, action, boundary, bounces):
    terminal = _net_fail(tracks, action, boundary, bounces)
    if not terminal:
        return None
    return Terminal(
        terminal.frame, terminal.timestamp, terminal.loser, terminal.winner,
        terminal.confidence,
        terminal.reason.replace("net fail by hitter", "near-net terminal fail", 1),
        terminal.miss_type,
    )

def _out_miss(tracks, action, bounces, boundary):
    for bounce in bounces:
        if not action.frame < bounce.frame < boundary:
            continue
        point = _nearest_point(tracks.ball_court, bounce.frame, window=2)
        if not point or point.confidence < 0.55 or point.frame not in tracks.valid_homography_frames:
            continue
        if (point.x < -OUT_MARGIN or point.x > 1.0 + OUT_MARGIN
                or point.y < -OUT_MARGIN or point.y > 1.0 + OUT_MARGIN):
            loser = player_number(action.player_id)
            return Terminal(
                bounce.frame, bounce.timestamp, loser, opponent(loser),
                min(bounce.confidence, point.confidence),
                "classified bounce outside court margin", "out",
            )
    return None

def _no_return_miss(tracks, action, boundary, bounces):
    points = _court_segment(tracks, action.frame + 1, boundary)
    reached = (
        [point for point in points if point.y <= 0.45]
        if action.player_id == "player_1" else
        [point for point in points if point.y >= 0.55]
    )
    if not reached:
        return None
    terminal_bounces = [bounce for bounce in bounces if reached[0].frame <= bounce.frame < boundary]
    evidence = terminal_bounces[-1] if terminal_bounces else reached[0]
    receiver = other_player(action.player_id)
    return Terminal(
        evidence.frame, evidence.timestamp, player_number(receiver),
        player_number(action.player_id), 0.78,
        "ball reached receiver side without valid return before clip ended", "no_return",
    )

def _terminal(tracks, action, boundary, bounces):
    return (
        _missed_receiver_net_fail(tracks, action, boundary, bounces)
        or _net_fail(tracks, action, boundary, bounces)
        or _near_net_terminal_fail(tracks, action, boundary, bounces)
        or _out_miss(tracks, action, bounces, boundary)
        or _no_return_miss(tracks, action, boundary, bounces)
    )

def _terminal_description(terminal):
    if terminal.miss_type == "net":
        return f"{terminal.loser} hit the net"
    if terminal.miss_type == "out":
        return f"{terminal.loser} hit out"
    return f"{terminal.loser} can't hit"

def build_events(tracks, episodes, bounces, reset_frames):
    segments, rejected = build_point_sequence(tracks, episodes, reset_frames, bounces)
    events, accepted, miss_types = [], [], Counter()
    missing_terminals = 0
    for segment in segments:
        contacts = list(segment.contacts)
        accepted.extend(contacts)
        terminal = _terminal(tracks, contacts[-1], segment.end_frame, bounces) if contacts else None
        for index, contact in enumerate(contacts):
            next_contact = contacts[index + 1] if index + 1 < len(contacts) else None
            end = next_contact.frame if next_contact else segment.end_frame
            reason = contact.reason
            if getattr(contact.episode, "debug_reason", ""):
                reason = f"{reason}; {contact.episode.debug_reason}"
            if contact.event_type == "hit":
                reason = (
                    f"{reason}; "
                    f"{_hit_direction_reason(tracks, contact, end - 1, bounces, next_contact, terminal)}"
                )
            events.append(Event(
                contact.frame, contact.timestamp, contact.event_type,
                player_number(contact.player_id), contact.confidence,
                _contact_description(tracks, contact, end - 1, bounces, next_contact, terminal), reason,
            ))
        if not terminal:
            missing_terminals += 1
            continue
        miss_types[terminal.miss_type] += 1
        terminal_type = "net" if terminal.miss_type == "net" else "miss"
        terminal_description = _terminal_description(terminal)
        events.append(Event(
            terminal.frame, terminal.timestamp, terminal_type, terminal.loser,
            terminal.confidence, terminal_description, terminal.reason,
        ))
        events.append(Event(
            terminal.frame, terminal.timestamp + 0.55, "point", terminal.winner,
            terminal.confidence, f"point to {terminal.winner}", terminal.reason,
        ))
    return BuildResult(
        tuple(events), tuple(accepted), tuple(rejected), missing_terminals, dict(miss_types),
    )
