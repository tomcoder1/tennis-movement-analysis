from pathlib import Path
from predictor import predictor
from homography import run_homography
from audio_description import extract_audio_description
from speech_scheduler import schedule_speech
from audio_commentary import generate_audio_commentary, mux_commentary_with_video
from draw import draw
from pipeline_utils import ensure_output_dirs, organized_path

base_output_dir = "outputs"

for video in sorted(Path("test").glob("test*.mp4")):
    case_name = video.stem
    video_path = str(video)
    output_dir = f"{base_output_dir}/{case_name}"

    ensure_output_dirs(output_dir)

    predictor(video_path, output_dir)
    run_homography(output_dir)
    extract_audio_description(output_dir)
    schedule_speech(output_dir, video_path=video_path)

    commentary_audio = generate_audio_commentary(output_dir, video_path)
    draw(video_path, output_dir)

    mux_commentary_with_video(
        organized_path(output_dir, "out_video"),
        commentary_audio,
        output_dir,
    )
