"""Optional text-to-speech output for spatial commentary."""

import csv
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import wave
from array import array

import cv2

from pipeline_utils import (
    ensure_output_dirs, organized_path, resolve_input_path, validate_csv,
)

_ONLINE_TTS_AVAILABLE = None


def _scheduled_rows(path):
    validate_csv(
        path,
        {"start_timestamp", "end_timestamp", "priority", "spoken_text", "was_skipped"},
        "speech-scheduler",
    )
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = []
        for row in csv.DictReader(handle):
            if str(row.get("was_skipped", "")).lower() == "true":
                continue
            description = row.get("spoken_text", "").strip()
            if not description:
                continue
            rows.append({
                "timestamp": float(row["start_timestamp"]),
                "end_timestamp": float(row["end_timestamp"]),
                "priority": int(row["priority"]),
                "description": description,
            })
    return rows


def _run_tts_subprocess(code, text, text_path, output_path, timeout):
    with open(text_path, "w", encoding="utf-8") as handle:
        handle.write(text)
    try:
        result = subprocess.run(
            [sys.executable, "-c", code, text_path, output_path],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    finally:
        if os.path.isfile(text_path):
            os.remove(text_path)
    if result.returncode != 0:
        message = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "engine failed"
        raise RuntimeError(message)
    if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError("engine did not create an audio file")
    return output_path


def _generate_pyttsx3(text, text_path, wav_path):
    code = (
        "import sys, pyttsx3; text=open(sys.argv[1], encoding='utf-8').read(); "
        "engine=pyttsx3.init(); engine.setProperty('rate', 155); "
        "engine.save_to_file(text, sys.argv[2]); engine.runAndWait()"
    )
    return _run_tts_subprocess(code, text, text_path, wav_path, 45)


def _online_tts_available():
    global _ONLINE_TTS_AVAILABLE
    if _ONLINE_TTS_AVAILABLE is not None:
        return _ONLINE_TTS_AVAILABLE
    try:
        import gtts  # noqa: F401
        with urllib.request.urlopen("https://translate.google.com", timeout=2):
            _ONLINE_TTS_AVAILABLE = True
    except (ImportError, OSError, ValueError):
        _ONLINE_TTS_AVAILABLE = False
    return _ONLINE_TTS_AVAILABLE


def _generate_gtts_clip(text, directory, index):
    if not _online_tts_available():
        raise RuntimeError("gTTS is offline")
    text_path = os.path.join(directory, f"clip_{index}.txt")
    mp3_path = os.path.join(directory, f"clip_{index}.mp3")
    wav_path = os.path.join(directory, f"clip_{index}.wav")
    code = (
        "import sys; from gtts import gTTS; "
        "text=open(sys.argv[1], encoding='utf-8').read(); "
        "gTTS(text=text, lang='en', slow=False).save(sys.argv[2])"
    )
    _run_tts_subprocess(code, text, text_path, mp3_path, 30)
    ffmpeg = _ffmpeg_executable()
    if not ffmpeg:
        raise RuntimeError("FFmpeg is unavailable for gTTS conversion")
    result = subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", mp3_path,
         "-ac", "1", "-ar", "22050", wav_path],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0 or not os.path.isfile(wav_path):
        raise RuntimeError("gTTS WAV conversion failed")
    return wav_path


def _generate_windows_speech(text, text_path, wav_path):
    powershell = shutil.which("powershell.exe") or shutil.which("powershell")
    if os.name != "nt" or not powershell:
        raise RuntimeError("Windows speech synthesis is unavailable")
    with open(text_path, "w", encoding="utf-8") as handle:
        handle.write(text)
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$voice = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$voice.Rate = 1; "
        "$voice.SetOutputToWaveFile($env:TTS_WAV_PATH); "
        "$voice.Speak([System.IO.File]::ReadAllText($env:TTS_TEXT_PATH)); "
        "$voice.Dispose()"
    )
    environment = os.environ.copy()
    environment["TTS_TEXT_PATH"] = os.path.abspath(text_path)
    environment["TTS_WAV_PATH"] = os.path.abspath(wav_path)
    try:
        result = subprocess.run(
            [powershell, "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            check=False,
            timeout=45,
            env=environment,
        )
    finally:
        if os.path.isfile(text_path):
            os.remove(text_path)
    if result.returncode != 0:
        message = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "engine failed"
        raise RuntimeError(message)
    if not os.path.isfile(wav_path) or os.path.getsize(wav_path) == 0:
        raise RuntimeError("Windows speech synthesis did not create a WAV file")
    return wav_path


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
    return mp3_path if result.returncode == 0 and os.path.isfile(mp3_path) else None


def _ffmpeg_executable():
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError, OSError):
        return None


def mux_commentary_with_video(video_path, audio_path, output_dir="outputs"):
    """Attach the synchronized description track to the annotated MP4."""
    if not audio_path or not os.path.isfile(audio_path):
        print("Described video skipped: synchronized commentary audio is unavailable.")
        return None
    if not os.path.isfile(video_path):
        print(f"Described video skipped: annotated video not found: {video_path}")
        return None
    ffmpeg = _ffmpeg_executable()
    if not ffmpeg:
        print("Described video skipped: FFmpeg is unavailable.")
        return None
    output_path = organized_path(output_dir, "described_video")
    result = subprocess.run(
        [
            ffmpeg, "-y", "-loglevel", "error",
            "-i", video_path, "-i", audio_path,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
            "-shortest", output_path,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not os.path.isfile(output_path):
        message = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "FFmpeg failed"
        print(f"Described video skipped: {message}")
        return None
    print(f"Saved synchronized described video to: {output_path}")
    return output_path


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


def _generate_local_clip(text, directory, index):
    text_path = os.path.join(directory, f"clip_{index}.txt")
    wav_path = os.path.join(directory, f"clip_{index}.wav")
    errors = []
    if _online_tts_available():
        try:
            return _generate_gtts_clip(text, directory, index)
        except Exception as exc:
            errors.append(f"gTTS: {exc}")
    engines = (
        (_generate_windows_speech, "Windows speech"),
        (_generate_pyttsx3, "pyttsx3"),
    ) if os.name == "nt" else (
        (_generate_pyttsx3, "pyttsx3"),
    )
    for engine, name in engines:
        try:
            return engine(text, text_path, wav_path)
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    raise RuntimeError("; ".join(errors))


def _trimmed_wave(path):
    with wave.open(path, "rb") as source:
        params = source.getparams()
        data = source.readframes(source.getnframes())
    if params.sampwidth != 2:
        return params, data
    samples = array("h")
    samples.frombytes(data)
    if sys.byteorder == "big":
        samples.byteswap()
    channels = params.nchannels
    active = []
    for frame in range(len(samples) // channels):
        start = frame * channels
        if max(abs(samples[start + channel]) for channel in range(channels)) > 180:
            active.append(frame)
    if not active:
        return params, data
    padding = int(params.framerate * 0.06)
    first = max(0, active[0] - padding)
    last = min(len(samples) // channels, active[-1] + padding)
    trimmed = samples[first * channels:last * channels]
    if sys.byteorder == "big":
        trimmed.byteswap()
    return params, trimmed.tobytes()


def _schedule_clips(groups, video_duration):
    scheduled = []
    last_end = 0.0
    for index, group in enumerate(groups):
        params, data = _trimmed_wave(group["clip"])
        clip_duration = len(data) / (params.framerate * params.sampwidth * params.nchannels)
        start = group["timestamp"]
        if start < last_end + 0.25:
            if group["priority"] >= 100 and scheduled and scheduled[-1]["priority"] < 100:
                scheduled.pop()
                last_end = scheduled[-1]["end"] if scheduled else 0.0
            else:
                continue
        next_start = groups[index + 1]["timestamp"] if index + 1 < len(groups) else video_duration
        hard_end = min(video_duration, next_start - 0.25)
        maximum_duration = hard_end - start
        if maximum_duration <= 0:
            continue
        if clip_duration > maximum_duration:
            if group["priority"] < 100:
                continue
            # Must-speak lines are already reduced to their shortest template.
            # As a final guard, sample them slightly faster instead of allowing lag.
            frame_size = params.sampwidth * params.nchannels
            input_frames = len(data) // frame_size
            output_frames = max(1, int(maximum_duration * params.framerate))
            compressed = bytearray(output_frames * frame_size)
            for output_frame in range(output_frames):
                input_frame = min(input_frames - 1, int(output_frame * input_frames / output_frames))
                source = input_frame * frame_size
                target = output_frame * frame_size
                compressed[target:target + frame_size] = data[source:source + frame_size]
            data = bytes(compressed)
            clip_duration = maximum_duration
        scheduled.append({
            "start": start,
            "end": start + clip_duration,
            "params": params,
            "data": data,
            "text": group["text"],
            "priority": group["priority"],
        })
        last_end = start + clip_duration
    return scheduled


def _write_timeline_wav(scheduled, video_duration, output_path):
    if not scheduled:
        raise RuntimeError("No commentary clips fit on the video timeline")
    first = scheduled[0]["params"]
    frame_size = first.sampwidth * first.nchannels
    total_frames = int(round(video_duration * first.framerate))
    timeline = bytearray(total_frames * frame_size)
    for clip in scheduled:
        params = clip["params"]
        if (params.nchannels, params.sampwidth, params.framerate) != (
            first.nchannels, first.sampwidth, first.framerate,
        ):
            raise RuntimeError("Speech clips used incompatible WAV formats")
        offset = int(round(clip["start"] * first.framerate)) * frame_size
        end = min(len(timeline), offset + len(clip["data"]))
        timeline[offset:end] = clip["data"][:end - offset]
    with wave.open(output_path, "wb") as target:
        target.setnchannels(first.nchannels)
        target.setsampwidth(first.sampwidth)
        target.setframerate(first.framerate)
        target.writeframes(timeline)


def _generate_synchronized_audio(rows, video_path, wav_path):
    video_duration = _video_duration(video_path)
    groups = [
        {
            "timestamp": row["timestamp"],
            "end_timestamp": row["end_timestamp"],
            "text": row["description"],
            "priority": 101 - row["priority"],
            "rows": [row],
        }
        for row in rows
    ]
    with tempfile.TemporaryDirectory(dir=os.path.dirname(wav_path)) as directory:
        generated = []
        for index, group in enumerate(groups):
            group["clip"] = _generate_local_clip(group["text"], directory, index)
            generated.append(group)
        scheduled = _schedule_clips(generated, video_duration)
        _write_timeline_wav(scheduled, video_duration, wav_path)
    print(f"Scheduled {len(scheduled)} commentary clips across {video_duration:.2f}s")
    return wav_path


def generate_audio_commentary(output_dir="outputs", video_path="test.mp4"):
    """Generate a timeline-aligned audio-description track."""
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

    mp3_path = organized_path(output_dir, "commentary_mp3")
    wav_path = organized_path(output_dir, "commentary_wav")
    try:
        path = _generate_synchronized_audio(rows, video_path, wav_path)
        converted = _convert_wav_to_mp3(path, mp3_path)
        print(f"Saved synchronized commentary to: {path}")
        if converted:
            print(f"Saved synchronized MP3 commentary to: {converted}")
            return converted
        return path
    except Exception as exc:
        print(f"Synchronized audio commentary unavailable: {exc}")
        return None


if __name__ == "__main__":
    generate_audio_commentary()
