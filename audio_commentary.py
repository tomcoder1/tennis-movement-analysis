import csv
import os
import shutil
import subprocess
import sys
import tempfile
import wave
from array import array
from pathlib import Path

import cv2

from pipeline_utils import (
    ensure_output_dirs,
    organized_path,
    resolve_input_path,
    validate_csv,
)

TTS_RATE = 260
TTS_VOLUME = 1.0
TTS_VOICE_CONTAINS = ""

def _scheduled_rows(path):
    validate_csv(
        path,
        {"start_timestamp", "end_timestamp", "moment_type", "text_spoken"},
        "speech-scheduler",
    )

    rows = []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            text = row.get("text_spoken", "").strip()
            if not text:
                continue

            moment_type = row.get("moment_type", "")
            rows.append(
                {
                    "timestamp": float(row["start_timestamp"]),
                    "end_timestamp": float(row["end_timestamp"]),
                    "priority": {
                        "point": 1,
                        "miss": 2,
                        "serve": 3,
                        "hit": 4,
                    }.get(moment_type, 4),
                    "text": text,
                }
            )

    return rows

def _ffmpeg_executable():
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg

    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError, OSError):
        return None

def _video_duration(video_path):
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video for audio timing: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    capture.release()

    if fps <= 0 or frames <= 0:
        raise RuntimeError(f"Could not determine video duration: {video_path}")

    return frames / fps

def _run_tts_subprocess(text, text_path, wav_path):
    with open(text_path, "w", encoding="utf-8") as handle:
        handle.write(text)

    code = """
import sys
import pyttsx3

text_path = sys.argv[1]
wav_path = sys.argv[2]
rate = int(sys.argv[3])
volume = float(sys.argv[4])
voice_contains = sys.argv[5].strip().lower()

with open(text_path, encoding="utf-8") as handle:
    text = handle.read()

engine = pyttsx3.init()
engine.setProperty("rate", rate)
engine.setProperty("volume", volume)

if voice_contains:
    for voice in engine.getProperty("voices"):
        name = getattr(voice, "name", "") or ""
        voice_id = getattr(voice, "id", "") or ""
        if voice_contains in name.lower() or voice_contains in voice_id.lower():
            engine.setProperty("voice", voice.id)
            break

engine.save_to_file(text, wav_path)
engine.runAndWait()
engine.stop()
"""
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                code,
                text_path,
                wav_path,
                str(TTS_RATE),
                str(TTS_VOLUME),
                TTS_VOICE_CONTAINS,
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=45,
        )
    finally:
        if os.path.isfile(text_path):
            os.remove(text_path)

    if result.returncode != 0:
        message = result.stderr.strip().splitlines()
        raise RuntimeError(message[-1] if message else "pyttsx3 failed.")

    if not os.path.isfile(wav_path) or os.path.getsize(wav_path) == 0:
        raise RuntimeError("pyttsx3 did not create a WAV file.")

    return wav_path

def _generate_clip(text, directory, index):
    text_path = os.path.join(directory, f"clip_{index}.txt")
    wav_path = os.path.join(directory, f"clip_{index}.wav")
    return _run_tts_subprocess(text, text_path, wav_path)

def _trimmed_wave(path):
    with wave.open(path, "rb") as source:
        params = source.getparams()
        data = source.readframes(source.getnframes())

    if params.sampwidth != 2:
        return params, data

    samples = array("h")
    samples.frombytes(data)

    channels = params.nchannels
    active_frames = []

    for frame in range(len(samples) // channels):
        start = frame * channels
        loudest = max(abs(samples[start + channel]) for channel in range(channels))
        if loudest > 180:
            active_frames.append(frame)

    if not active_frames:
        return params, data

    padding = int(params.framerate * 0.04)
    first = max(0, active_frames[0] - padding)
    last = min(len(samples) // channels, active_frames[-1] + padding)

    trimmed = samples[first * channels:last * channels]
    return params, trimmed.tobytes()

def _schedule_clips(groups, video_duration):
    scheduled = []
    last_end = 0.0

    for group in groups:
        params, data = _trimmed_wave(group["clip"])
        clip_duration = len(data) / (
            params.framerate * params.sampwidth * params.nchannels
        )

        start = max(group["timestamp"], last_end + 0.08)

        if start + clip_duration > video_duration:
            if group["priority"] == 4:
                continue
            start = max(last_end + 0.08, video_duration - clip_duration)

        scheduled.append(
            {
                "start": start,
                "end": start + clip_duration,
                "params": params,
                "data": data,
                "text": group["text"],
                "priority": group["priority"],
            }
        )

        last_end = start + clip_duration

    return scheduled

def _write_timeline_wav(scheduled, video_duration, output_path):
    if not scheduled:
        raise RuntimeError("No commentary clips fit on the video timeline.")

    first = scheduled[0]["params"]
    frame_size = first.sampwidth * first.nchannels
    total_frames = int(round(video_duration * first.framerate))
    timeline = bytearray(total_frames * frame_size)

    for clip in scheduled:
        params = clip["params"]

        if (params.nchannels, params.sampwidth, params.framerate) != (
            first.nchannels,
            first.sampwidth,
            first.framerate,
        ):
            raise RuntimeError("Speech clips used incompatible WAV formats.")

        offset = int(round(clip["start"] * first.framerate)) * frame_size
        end = min(len(timeline), offset + len(clip["data"]))
        timeline[offset:end] = clip["data"][:end - offset]

    with wave.open(output_path, "wb") as target:
        target.setnchannels(first.nchannels)
        target.setsampwidth(first.sampwidth)
        target.setframerate(first.framerate)
        target.writeframes(timeline)

def _convert_wav_to_mp3(wav_path, mp3_path):
    ffmpeg = _ffmpeg_executable()
    if not ffmpeg:
        return None

    result = subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", wav_path, mp3_path],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode == 0 and os.path.isfile(mp3_path):
        return mp3_path

    return None

def _generate_synchronized_audio(rows, video_path, wav_path):
    video_duration = _video_duration(video_path)

    groups = [
        {
            "timestamp": row["timestamp"],
            "end_timestamp": row["end_timestamp"],
            "text": row["text"],
            "priority": row["priority"],
        }
        for row in rows
    ]

    with tempfile.TemporaryDirectory(dir=os.path.dirname(wav_path)) as directory:
        generated = []

        for index, group in enumerate(groups):
            group["clip"] = _generate_clip(group["text"], directory, index)
            generated.append(group)

        scheduled = _schedule_clips(generated, video_duration)
        _write_timeline_wav(scheduled, video_duration, wav_path)

    print("Used TTS engine: pyttsx3")
    print(f"Used TTS rate: {TTS_RATE}")
    print(f"Scheduled {len(scheduled)} commentary clips across {video_duration:.2f}s")
    return wav_path

def _commentary_srt_path(output_dir):
    try:
        return resolve_input_path(output_dir, "commentary_srt")
    except Exception:
        return os.path.join(output_dir, "narration", "commentary.srt")


def _subtitle_filter_path(path):
    return Path(path).resolve().as_posix().replace(":", "\\:")

def generate_audio_commentary(output_dir="outputs", video_path="test.mp4"):
    ensure_output_dirs(output_dir)

    csv_path = resolve_input_path(output_dir, "scheduled_lines")

    try:
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(
                f"Missing {organized_path(output_dir, 'scheduled_lines')}. "
                "Run the speech scheduler first."
            )

        rows = _scheduled_rows(csv_path)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Audio commentary skipped: {exc}")
        return None

    if not rows:
        print("Audio commentary skipped: no descriptions were available.")
        return None

    wav_path = organized_path(output_dir, "commentary_wav")
    mp3_path = organized_path(output_dir, "commentary_mp3")

    try:
        generated_wav = _generate_synchronized_audio(rows, video_path, wav_path)
        generated_mp3 = _convert_wav_to_mp3(generated_wav, mp3_path)

        print(f"Saved synchronized commentary to: {generated_wav}")

        if generated_mp3:
            print(f"Saved synchronized MP3 commentary to: {generated_mp3}")
            return generated_mp3

        return generated_wav

    except Exception as exc:
        print(f"Synchronized audio commentary unavailable: {exc}")
        return None

def mux_commentary_with_video(video_path, audio_path, output_dir="outputs"):
    if not audio_path or not os.path.isfile(audio_path):
        print("Described video skipped: commentary audio is unavailable.")
        return None

    if not os.path.isfile(video_path):
        print(f"Described video skipped: annotated video not found: {video_path}")
        return None

    ffmpeg = _ffmpeg_executable()
    if not ffmpeg:
        print("Described video skipped: FFmpeg is unavailable.")
        return None

    output_path = organized_path(output_dir, "described_video")
    srt_path = _commentary_srt_path(output_dir)

    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        video_path,
        "-i",
        audio_path,
    ]

    if os.path.isfile(srt_path):
        command.extend(
            [
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "libx264",
                "-vf",
                f"subtitles='{_subtitle_filter_path(srt_path)}'",
                "-c:a",
                "aac",
                "-b:a",
                "160k",
                "-shortest",
                output_path,
            ]
        )
    else:
        command.extend(
            [
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "160k",
                "-shortest",
                output_path,
            ]
        )

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0 or not os.path.isfile(output_path):
        message = result.stderr.strip().splitlines()
        print(f"Described video skipped: {message[-1] if message else 'FFmpeg failed'}")
        return None

    if os.path.isfile(srt_path):
        print(f"Saved described video with subtitles to: {output_path}")
    else:
        print(f"Saved described video without subtitles to: {output_path}")

    return output_path

if __name__ == "__main__":
    generate_audio_commentary()