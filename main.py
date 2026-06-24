from predictor import predictor
from homography import run_homography
from audio_description import extract_audio_description
from speech_scheduler import schedule_speech
from audio_commentary import generate_audio_commentary, mux_commentary_with_video
from draw import draw
from pipeline_utils import ensure_output_dirs, organized_path

VIDEO_PATH = "test.mp4"
OUTPUT_DIR = "outputs"

ensure_output_dirs(OUTPUT_DIR)

predictor(VIDEO_PATH, OUTPUT_DIR)
run_homography(OUTPUT_DIR)
extract_audio_description(OUTPUT_DIR)
schedule_speech(OUTPUT_DIR, video_path=VIDEO_PATH)
commentary_audio = generate_audio_commentary(OUTPUT_DIR, VIDEO_PATH)
draw(VIDEO_PATH, OUTPUT_DIR)
mux_commentary_with_video(
    organized_path(OUTPUT_DIR, "out_video"),
    commentary_audio,
    OUTPUT_DIR,
)
