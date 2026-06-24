from predictor import predictor
from homography import run_homography
from ball_events import detect_ball_events
from tennis_events import interpret_tennis_events
from draw import draw

VIDEO_PATH = "test.mp4"

predictor(VIDEO_PATH)
run_homography()
detect_ball_events()
interpret_tennis_events()
draw(VIDEO_PATH)
