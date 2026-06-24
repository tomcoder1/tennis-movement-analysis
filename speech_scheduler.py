"""Select short, non-overlapping narration lines from description moments."""

import csv

import cv2

from pipeline_utils import ensure_output_dirs, organized_path, parse_number, validate_csv


WORDS_PER_SECOND = 2.4
MIN_GAP_SECONDS = 0.25

SCHEDULE_FIELDS = [
    "line_id", "moment_id", "start_timestamp", "end_timestamp", "priority",
    "available_seconds", "estimated_spoken_seconds", "text_source",
    "spoken_text", "was_shortened", "was_skipped", "skip_reason",
]


def estimate_speech_seconds(text):
    return len(text.split()) / WORDS_PER_SECOND if text else 0.0


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
        row["importance"] = int(parse_number(row.get("importance")) or 3)
        row["confidence"] = parse_number(row.get("confidence")) or 0.0
    return sorted(rows, key=lambda row: row["start_timestamp"])


def _word_limit(available):
    if available < 0.8:
        return 3
    if available < 1.4:
        return 3
    if available < 2.2:
        return 5
    if available < 3.2:
        return 8
    return 12


def choose_text(moment, available_seconds, mode="live"):
    if available_seconds < 0.8 and moment["importance"] != 1:
        return None, "", "insufficient time"
    maximum_words = _word_limit(available_seconds)
    if mode == "replay":
        choices = (
            ("description_long", moment.get("description_long", "")),
            ("description_short", moment.get("description_short", "")),
            ("description_tiny", moment.get("description_tiny", "")),
        )
    elif available_seconds < 1.4:
        choices = (("description_tiny", moment.get("description_tiny", "")),)
    elif available_seconds < 3.2:
        choices = (
            ("description_short", moment.get("description_short", "")),
            ("description_tiny", moment.get("description_tiny", "")),
        )
    else:
        choices = (
            ("description_long", moment.get("description_long", "")),
            ("description_short", moment.get("description_short", "")),
            ("description_tiny", moment.get("description_tiny", "")),
        )
    for source, text in choices:
        text = text.strip()
        if not text or len(text.split()) > maximum_words:
            continue
        if estimate_speech_seconds(text) <= available_seconds:
            return text, source, ""
    return None, "", "no description version fits"


def _combine_related_moments(moments):
    skipped = {}
    for index, moment in enumerate(moments[:-1]):
        later = moments[index + 1]
        if (moment["moment_type"] == "serve_start" and later["moment_type"] == "serve_result"
                and later["start_timestamp"] - moment["start_timestamp"] <= 1.2):
            skipped[moment["moment_id"]] = "combined with serve result"
    by_time = {}
    for moment in moments:
        by_time.setdefault(round(moment["start_timestamp"], 2), []).append(moment)
    for group in by_time.values():
        miss = next((item for item in group if item["moment_type"] == "miss"), None)
        point = next((item for item in group if item["moment_type"] == "point_result"), None)
        if miss and point:
            skipped[miss["moment_id"]] = "combined with point result"
            point["description_long"] = f'{miss["description_short"]} {point["description_long"]}'
            point["description_short"] = f'{miss["description_tiny"]} {point["description_short"]}'
    return skipped


def _next_important_time(moments, index, skipped, timeline_end):
    current = moments[index]["start_timestamp"]
    for later in moments[index + 1:]:
        if later["moment_id"] in skipped:
            continue
        if later["importance"] <= 2 and later["start_timestamp"] > current + 0.01:
            return later["start_timestamp"]
    return timeline_end


def _write_outputs(lines, output_dir):
    csv_path = organized_path(output_dir, "scheduled_lines")
    text_path = organized_path(output_dir, "commentary_text")
    srt_path = organized_path(output_dir, "commentary_srt")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCHEDULE_FIELDS)
        writer.writeheader()
        writer.writerows(lines)
    spoken = [line for line in lines if line["was_skipped"] == "False"]
    with open(text_path, "w", encoding="utf-8") as handle:
        for line in spoken:
            handle.write(line["spoken_text"] + "\n")
    with open(srt_path, "w", encoding="utf-8") as handle:
        for index, line in enumerate(spoken, 1):
            start, end = float(line["start_timestamp"]), float(line["end_timestamp"])
            if index < len(spoken):
                end = min(end, float(spoken[index]["start_timestamp"]) - 0.01)
            handle.write(f"{index}\n")
            handle.write(f"{format_srt_timestamp(start)} --> {format_srt_timestamp(max(start + 0.1, end))}\n")
            handle.write(line["spoken_text"] + "\n\n")
    return csv_path


def schedule_speech(output_dir="outputs", video_path="test.mp4", mode="live", timeline_end=None):
    ensure_output_dirs(output_dir)
    moments_path = organized_path(output_dir, "moments")
    validate_csv(
        moments_path,
        {"moment_id", "start_timestamp", "moment_type", "importance", "confidence",
         "description_long", "description_short", "description_tiny"},
        "audio-description",
    )
    moments = _read_moments(moments_path)
    timeline_end = timeline_end or _video_duration(video_path) or (
        (moments[-1]["start_timestamp"] + 3.0) if moments else 0.0
    )
    combined = _combine_related_moments(moments)
    lines = []
    speech_free_at = 0.0
    previous_text = ""
    previous_text_time = -10_000.0

    for index, moment in enumerate(moments):
        next_time = _next_important_time(moments, index, combined, timeline_end)
        available = max(0.0, next_time - moment["start_timestamp"] - MIN_GAP_SECONDS)
        line = {
            "line_id": f"L{index + 1:04d}", "moment_id": moment["moment_id"],
            "start_timestamp": f'{moment["start_timestamp"]:.6f}', "end_timestamp": "",
            "priority": moment["importance"], "available_seconds": f"{available:.3f}",
            "estimated_spoken_seconds": "0.000", "text_source": "", "spoken_text": "",
            "was_shortened": "False", "was_skipped": "True", "skip_reason": "",
        }
        if moment["moment_id"] in combined:
            line["skip_reason"] = combined[moment["moment_id"]]
        elif moment["importance"] == 3 and mode == "live":
            line["skip_reason"] = "low-priority moment"
        elif moment["confidence"] < 0.35 and moment["importance"] > 1:
            line["skip_reason"] = "low confidence"
        elif moment["start_timestamp"] < speech_free_at + MIN_GAP_SECONDS:
            line["skip_reason"] = "would overlap previous speech"
        else:
            text, source, reason = choose_text(moment, available, mode)
            if not text:
                line["skip_reason"] = reason
            elif text == previous_text and moment["start_timestamp"] - previous_text_time < 2.0:
                line["skip_reason"] = "duplicate line"
            else:
                spoken_seconds = estimate_speech_seconds(text)
                end = moment["start_timestamp"] + spoken_seconds
                line.update({
                    "end_timestamp": f"{end:.6f}", "estimated_spoken_seconds": f"{spoken_seconds:.3f}",
                    "text_source": source, "spoken_text": text,
                    "was_shortened": str(source != "description_long"),
                    "was_skipped": "False", "skip_reason": "",
                })
                speech_free_at, previous_text = end, text
                previous_text_time = moment["start_timestamp"]
        lines.append(line)

    output_path = _write_outputs(lines, output_dir)
    count = sum(line["was_skipped"] == "False" for line in lines)
    print(f"Scheduled {count}/{len(lines)} narration lines: {output_path}")
    return lines


if __name__ == "__main__":
    schedule_speech()
