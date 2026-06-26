import csv

from bounce_detection import BounceDetector
from contact_detection import detect_clip_boundaries, detect_contact_episodes
from pipeline_utils import ensure_output_dirs, organized_path
from point_builder import build_events, player_number
from track_loading import load_tracks

MOMENT_FIELDS = [
    "moment_id", "start_timestamp", "end_timestamp", "moment_type",
    "player_id", "description", "confidence", "reason", "source_event_id",
]
CONTACT_FIELDS = [
    "episode_id", "start_frame", "end_frame", "best_frame", "timestamp",
    "player", "confidence", "source", "precheck_reason", "debug_reason",
    "proximity", "speed_before", "speed_after", "court_proximity",
    "court_direction_change",
]
BOUNCE_FIELDS = [
    "frame_id", "timestamp", "court_x", "court_y", "confidence", "valid",
]
CONTACT_DEBUG_FIELDS = [
    "source_event_id", "frame_id", "timestamp", "player_id", "accepted",
    "event_type", "confidence", "reason", "contact_source", "ball_state",
    "nearest_player", "distance_to_player", "speed_before", "speed_after",
    "direction_change", "court_x", "court_y", "court_proximity",
    "court_direction_change", "homography_status", "player_identity_source",
]

def _write(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def _nearest_court_point(points, frame, window=2):
    nearby = [point for point in points if abs(point.frame - frame) <= window]
    return min(nearby, key=lambda point: abs(point.frame - frame)) if nearby else None

def extract_audio_description(output_dir="outputs"):
    ensure_output_dirs(output_dir)
    tracks = load_tracks(output_dir)
    boundaries = detect_clip_boundaries(tracks.player_court)
    episodes = detect_contact_episodes(tracks, boundaries)
    bounces = BounceDetector().predict(tracks.image_ball_observed)
    result = build_events(tracks, episodes, bounces, boundaries)

    moments = [{
        "moment_id": f"M{index:04d}",
        "start_timestamp": f"{event.timestamp:.6f}",
        "end_timestamp": f"{event.timestamp + 0.2:.6f}",
        "moment_type": event.event_type,
        "player_id": event.player,
        "description": event.description,
        "confidence": f"{event.confidence:.6f}",
        "reason": event.source_reason,
        "source_event_id": f"E{index:04d}",
    } for index, event in enumerate(result.events, 1)]
    _write(organized_path(output_dir, "moments"), MOMENT_FIELDS, moments)

    contacts = [{
        "episode_id": f"C{index:04d}",
        "start_frame": episode.start_frame or episode.frame,
        "end_frame": episode.end_frame or episode.frame,
        "best_frame": episode.frame,
        "timestamp": f"{episode.timestamp:.6f}",
        "player": player_number(episode.player_id),
        "confidence": f"{episode.confidence:.6f}",
        "source": getattr(episode, "contact_source", "image"),
        "precheck_reason": getattr(episode, "precheck_reason", ""),
        "debug_reason": getattr(episode, "debug_reason", ""),
        "proximity": f"{getattr(episode, 'proximity', 0.0):.6f}",
        "speed_before": f"{getattr(episode, 'speed_before', 0.0):.6f}",
        "speed_after": f"{getattr(episode, 'speed_after', 0.0):.6f}",
        "court_proximity": f"{getattr(episode, 'court_proximity', 0.0):.6f}",
        "court_direction_change": f"{getattr(episode, 'court_direction_change', 0.0):.6f}",
    } for index, episode in enumerate(episodes, 1)]
    _write(organized_path(output_dir, "contact_episodes"), CONTACT_FIELDS, contacts)

    accepted_by_episode = {
        id(contact.episode): contact for contact in result.accepted
    }
    debug_rows = []
    for index, episode in enumerate(episodes, 1):
        accepted = accepted_by_episode.get(id(episode))
        rejected = next((item for item in result.rejected if item.episode is episode), None)
        court_point = tracks.ball_court_by_frame.get(episode.frame)
        debug_rows.append({
            "source_event_id": f"C{index:04d}",
            "frame_id": episode.frame,
            "timestamp": f"{episode.timestamp:.6f}",
            "player_id": player_number(episode.player_id),
            "accepted": "1" if accepted else "0",
            "event_type": accepted.event_type if accepted else "",
            "confidence": f"{episode.confidence:.6f}",
            "reason": accepted.reason if accepted else (rejected.reason if rejected else "not selected"),
            "contact_source": getattr(episode, "contact_source", "image"),
            "ball_state": tracks.ball_state.get(episode.frame, "unknown"),
            "nearest_player": player_number(episode.player_id),
            "distance_to_player": f"{getattr(episode, 'proximity', 0.0):.6f}",
            "speed_before": f"{getattr(episode, 'speed_before', 0.0):.6f}",
            "speed_after": f"{getattr(episode, 'speed_after', 0.0):.6f}",
            "direction_change": f"{getattr(episode, 'impulse', 0.0):.6f}",
            "court_x": f"{court_point.x:.6f}" if court_point else "",
            "court_y": f"{court_point.y:.6f}" if court_point else "",
            "court_proximity": f"{getattr(episode, 'court_proximity', 0.0):.6f}",
            "court_direction_change": f"{getattr(episode, 'court_direction_change', 0.0):.6f}",
            "homography_status": "valid" if episode.frame in tracks.valid_homography_frames else "missing",
            "player_identity_source": "persistent_id",
        })
    _write(organized_path(output_dir, "contact_debug"), CONTACT_DEBUG_FIELDS, debug_rows)

    bounce_rows = []
    for bounce in bounces:
        point = _nearest_court_point(tracks.ball_court, bounce.frame)
        valid = bool(point and point.frame in tracks.valid_homography_frames)
        bounce_rows.append({
            "frame_id": bounce.frame,
            "timestamp": f"{bounce.timestamp:.6f}",
            "court_x": f"{point.x:.6f}" if point else "",
            "court_y": f"{point.y:.6f}" if point else "",
            "confidence": f"{bounce.confidence:.6f}",
            "valid": str(valid),
        })
    _write(organized_path(output_dir, "bounce_events"), BOUNCE_FIELDS, bounce_rows)
    print(f"Saved {len(moments)} moments, {len(contacts)} contacts, and {len(bounce_rows)} bounces")
    return moments

if __name__ == "__main__":
    extract_audio_description()
