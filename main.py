from predictor import predictor
from homography import run_homography
from draw import draw

VIDEO_PATH = "test.mp4"

predictor(VIDEO_PATH)
run_homography()
draw(VIDEO_PATH)