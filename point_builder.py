import math
from collections import Counter
from dataclasses import dataclass
from statistics import median

MIN_CONTACT_SECONDS = 0.18
DUPLICATE_SECONDS = 1.0
OUT_MARGIN = 0.03
NET_ZONE = 0.09
DEAD_GAP_FRAMES = 12

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

def _validate_segment(tracks, episodes):
    accepted, rejected = [], []
    expected_hitter = None
    for episode in episodes:
        if episode.precheck_reason:
            rejected.append(RejectedContact(episode, episode.precheck_reason))
            continue
        if not accepted:
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
            reason = "contact by expected receiver"
        expected_hitter = other_player(episode.player_id)
        accepted.append(AcceptedContact(episode, "hit", expected_hitter, reason))
    return accepted, rejected

def build_point_sequence(tracks, episodes, reset_frames, bounces=()):
    boundaries = [0, *sorted(reset_frames), tracks.last_frame + 1]
    segments, rejected = [], []
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:]), 1):
        candidates = [episode for episode in episodes if start <= episode.frame < end]
        overlaps = [
            episode for episode in candidates
            if any(bounce.confidence >= 0.80 and abs(bounce.frame - episode.frame) <= 2 for bounce in bounces)
        ]
        candidates = [episode for episode in candidates if episode not in overlaps]
        accepted, invalid = _validate_segment(tracks, candidates)
        segments.append(PointSegment(index, start, end, tuple(accepted)))
        rejected.extend(invalid)
        rejected.extend(RejectedContact(episode, "classified bounce overlap") for episode in overlaps)
    return segments, rejected

def _source_x(tracks, contact):
    source = _nearest_point(tracks.player_court.get(contact.player_id, []), contact.frame)
    return source.x if source and source.confidence >= 0.55 else 0.5

def _serve_target_x(tracks, contact_frame, end_frame, bounces):
    for bounce in bounces:
        if contact_frame < bounce.frame <= end_frame:
            point = _nearest_point(tracks.ball_court, bounce.frame, window=2)
            if point and point.confidence >= 0.55:
                return point.x
    points = [
        point for point in tracks.ball_court
        if contact_frame + 3 <= point.frame <= min(contact_frame + 14, end_frame)
        and point.confidence >= 0.55
    ]
    return median(point.x for point in points) if len(points) >= 3 else None

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
    elif abs(angle) <= 10.0:
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

def _net_miss(tracks, action, boundary):
    points = _court_segment(tracks, action.frame + 1, boundary)
    if len(points) < 5:
        return None
    from_bottom = action.player_id == "player_1"
    reached = [point for point in points if abs(point.y - 0.5) <= NET_ZONE]
    crossed = any(
        point.y < 0.5 - NET_ZONE if from_bottom else point.y > 0.5 + NET_ZONE
        for point in points
    )
    if not reached or crossed:
        return None
    near = reached[-1]
    loser = player_number(action.player_id)
    return Terminal(
        near.frame, near.timestamp, loser, opponent(loser), 0.82,
        "ball reached net zone without crossing before clip ended", "net",
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
        _net_miss(tracks, action, boundary)
        or _out_miss(tracks, action, bounces, boundary)
        or _no_return_miss(tracks, action, boundary, bounces)
    )

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
        events.append(Event(
            terminal.frame, terminal.timestamp, "miss", terminal.loser,
            terminal.confidence, f"{terminal.loser} miss", terminal.reason,
        ))
        events.append(Event(
            terminal.frame, terminal.timestamp + 0.55, "point", terminal.winner,
            terminal.confidence, f"point to {terminal.winner}", terminal.reason,
        ))
    return BuildResult(
        tuple(events), tuple(accepted), tuple(rejected), missing_terminals, dict(miss_types),
    )