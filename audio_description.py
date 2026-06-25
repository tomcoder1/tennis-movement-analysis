import csv

from bounce_detection import BounceDetector
from contact_detection import detect_clip_boundaries, detect_contact_episodes
from pipeline_utils import ensure_output_dirs, organized_path
from point_builder import build_events, player_number
from track_loading import load_tracks

MOMENT_FIELDS = [
    "moment_id", "start_timestamp", "end_timestamp", "moment_type",
    "player_id", "description",
]
CONTACT_FIELDS = [
    "episode_id", "start_frame", "end_frame", "best_frame", "timestamp",
    "player",
]
BOUNCE_FIELDS = [
    "frame_id", "timestamp", "court_x", "court_y", "confidence", "valid",
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
    bounces = BounceDetector().predict(tracks.image_ball)
    result = build_events(tracks, episodes, bounces, boundaries)

    moments = [{
        "moment_id": f"M{index:04d}",
        "start_timestamp": f"{event.timestamp:.6f}",
        "end_timestamp": f"{event.timestamp + 0.2:.6f}",
        "moment_type": event.event_type,
        "player_id": event.player,
        "description": event.description,
    } for index, event in enumerate(result.events, 1)]
    _write(organized_path(output_dir, "moments"), MOMENT_FIELDS, moments)

    contacts = [{
        "episode_id": f"C{index:04d}",
        "start_frame": episode.start_frame or episode.frame,
        "end_frame": episode.end_frame or episode.frame,
        "best_frame": episode.frame,
        "timestamp": f"{episode.timestamp:.6f}",
        "player": player_number(episode.player_id),
    } for index, episode in enumerate(episodes, 1)]
    _write(organized_path(output_dir, "contact_episodes"), CONTACT_FIELDS, contacts)

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