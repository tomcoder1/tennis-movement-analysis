import csv
import re

import cv2

from pipeline_utils import ensure_output_dirs, organized_path, parse_number, validate_csv

MIN_GAP_SECONDS = 0.12
EVENT_DURATION_SECONDS = {"serve": 0.85, "hit": 0.55, "miss": 0.45, "point": 0.60}
ALLOWED_LINE = re.compile(
    r"^(?:[12] hit(?: (?:left|right|straight))?|"
    r"[12] serve from (?:left|right|middle) to (?:left|right|middle)|"
    r"[12] miss|point to [12])$"
)
SCHEDULE_FIELDS = [
    "line_id", "start_timestamp", "end_timestamp", "moment_type", "player_id", "text_spoken",
]

def is_allowed_line(text):
    return bool(ALLOWED_LINE.fullmatch((text or "").strip().lower()))

def format_srt_timestamp(seconds):
    milliseconds = max(0, int(round(float(seconds) * 1000)))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def _video_duration(video_path):
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return None
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    capture.release()
    return frames / fps if fps > 0 and frames > 0 else None

def _read_moments(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["start_timestamp"] = parse_number(row.get("start_timestamp")) or 0.0
    return sorted(rows, key=lambda row: row["start_timestamp"])

def _write_outputs(lines, output_dir):
    csv_path = organized_path(output_dir, "scheduled_lines")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCHEDULE_FIELDS)
        writer.writeheader()
        writer.writerows(lines)
    spoken = [line for line in lines if line["text_spoken"]]
    with open(organized_path(output_dir, "commentary_text"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(line["text_spoken"] for line in spoken))
        if spoken:
            handle.write("\n")
    with open(organized_path(output_dir, "commentary_srt"), "w", encoding="utf-8") as handle:
        for index, line in enumerate(spoken, 1):
            handle.write(f"{index}\n")
            handle.write(
                f"{format_srt_timestamp(line['start_timestamp'])} --> "
                f"{format_srt_timestamp(line['end_timestamp'])}\n"
            )
            handle.write(line["text_spoken"] + "\n\n")
    return csv_path

def schedule_speech(output_dir="outputs", video_path="test.mp4", timeline_end=None):
    ensure_output_dirs(output_dir)
    moments_path = organized_path(output_dir, "moments")
    validate_csv(
        moments_path,
        {"start_timestamp", "moment_type", "player_id", "description"},
        "audio-description",
    )
    moments = _read_moments(moments_path)
    timeline_end = timeline_end or _video_duration(video_path) or (
        moments[-1]["start_timestamp"] + 3.0 if moments else 0.0
    )
    lines, free_at = [], 0.0
    for index, moment in enumerate(moments, 1):
        kind = moment.get("moment_type", "")
        text = moment.get("description", "").strip()
        start = max(moment["start_timestamp"], free_at + MIN_GAP_SECONDS)
        end = start + EVENT_DURATION_SECONDS.get(kind, 0.55)
        spoken = text if is_allowed_line(text) and (kind != "hit" or end <= timeline_end) else ""
        if spoken:
            free_at = end
        lines.append({
            "line_id": f"L{index:04d}",
            "start_timestamp": f"{start:.6f}" if spoken else "",
            "end_timestamp": f"{end:.6f}" if spoken else "",
            "moment_type": kind,
            "player_id": moment.get("player_id", ""),
            "text_spoken": spoken,
        })
    path = _write_outputs(lines, output_dir)
    print(f"Scheduled {sum(bool(line['text_spoken']) for line in lines)}/{len(lines)} events: {path}")
    return lines

if __name__ == "__main__":
    schedule_speech()