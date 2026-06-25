import csv
import math
import os

DETECTION_COLUMNS = {
    "frame_id", "timestamp", "source", "object_type", "object_id",
    "x", "y", "w", "h", "confidence", "track_id",
}

OUTPUT_LAYOUT = {
    "ball": ("raw", "ball.csv"),
    "player": ("raw", "player.csv"),
    "court": ("raw", "court.csv"),
    "homography": ("geometry", "homography.csv"),
    "ball_homography": ("geometry", "ball_homography.csv"),
    "player_homography": ("geometry", "player_homography.csv"),
    "contact_episodes": ("narration", "contact_episodes.csv"),
    "bounce_events": ("narration", "bounce.csv"),
    "moments": ("narration", "moments.csv"),
    "scheduled_lines": ("narration", "scheduled_lines.csv"),
    "out_video": ("media", "out.mp4"),
    "described_video": ("media", "out_described.mp4"),
    "commentary_text": ("narration", "commentary.txt"),
    "commentary_srt": ("narration", "commentary.srt"),
    "diagnosis_report": ("narration", "diagnosis_report.md"),
    "commentary_mp3": ("media", "commentary.mp3"),
    "commentary_wav": ("media", "commentary.wav"),
}

def organized_path(output_dir, key):
    try:
        folder, filename = OUTPUT_LAYOUT[key]
    except KeyError as exc:
        raise KeyError(f"Unknown pipeline output key: {key}") from exc
    return os.path.join(os.fspath(output_dir), folder, filename)

def resolve_input_path(output_dir, key):
    """Prefer organized input paths, falling back to legacy flat outputs."""
    preferred = organized_path(output_dir, key)
    if os.path.isfile(preferred):
        return preferred
    return os.path.join(os.fspath(output_dir), OUTPUT_LAYOUT[key][1])


def ensure_output_dirs(output_dir):
    for folder in ("raw", "geometry", "narration", "media"):
        os.makedirs(os.path.join(os.fspath(output_dir), folder), exist_ok=True)


def parse_number(value):
    """Return a finite float, or ``None`` for blank/invalid input."""
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def validate_video(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Input video not found: {path}. Set VIDEO_PATH in main.py."
        )


def validate_csv(path, required_columns, previous_stage=None):
    if not os.path.isfile(path):
        hint = f" Run the {previous_stage} stage first." if previous_stage else ""
        raise FileNotFoundError(f"Required CSV not found: {path}.{hint}")
    with open(path, newline="", encoding="utf-8-sig") as handle:
        columns = set(csv.DictReader(handle).fieldnames or [])
    missing = sorted(set(required_columns) - columns)
    if missing:
        hint = f" Re-run the {previous_stage} stage." if previous_stage else ""
        raise ValueError(
            f"CSV {path} is missing required columns: {', '.join(missing)}.{hint}"
        )


def require_files(paths, label):
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            f"Missing required {label}: {', '.join(missing)}. See README.md for setup."
        )
