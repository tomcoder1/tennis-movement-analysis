# Computer Vision Tennis Audio Description

This project turns a fixed-camera tennis video into short spatial narration for accessibility. It is not a television commentator or tennis referee. It describes a few useful moments—where players are, where the ball travels, and how a point appears to end—then leaves silence when extra speech would lag behind play.

## Pipeline

```text
video
  -> ball, player, and court tracking
  -> normalized court geometry
  -> audio-description moments
  -> live speech scheduling
  -> text-to-speech
  -> annotated and described video
```

The pipeline is deliberately small:

1. `predictor.py`
2. `homography.py`
3. `audio_description.py`
4. `speech_scheduler.py`
5. `audio_commentary.py`
6. `draw.py`

The old ball-event and tennis-event classifiers are not part of the system.

## Setup

Python 3.11–3.13 is supported.

```bash
uv sync
```

Required pretrained files:

| Path | Purpose |
| --- | --- |
| `model/model_best.pt` | ball heatmap model |
| `model/Court_detect_model.pth` | court keypoint model |
| `model/player_detector_best.pt` | player detector |
| `bytetrack.yaml` | player tracker configuration |

CUDA is used when available. CPU execution remains supported but slower.

## Run

Set `VIDEO_PATH` near the top of `main.py`, then run:

```bash
python main.py
```

There is no argparse setup and no required terminal flag.

## Outputs

```text
outputs/
  raw/
    ball.csv
    player.csv
    court.csv
  geometry/
    homography.csv
    ball_homography.csv
    player_homography.csv
  narration/
    moments.csv
    scheduled_lines.csv
    commentary.txt
    commentary.srt
  media/
    commentary.wav
    commentary.mp3
    out.mp4
    out_described.mp4
```

### Raw and geometry

- `ball.csv` stores reliable image-space ball observations; blank coordinates represent missing detections.
- `player.csv` stores player boxes and foot points.
- `court.csv` stores court keypoints.
- `homography.csv` stores image-to-court transforms.
- Ball and player homography CSVs store normalized court positions. Coordinates outside `0..1` are retained.

### Narration

- `moments.csv` contains useful spatial moments with long, short, and tiny neutral descriptions.
- `scheduled_lines.csv` records selected wording, timing budgets, and skip reasons.
- `commentary.txt` contains only spoken lines.
- `commentary.srt` uses the same scheduled text and timing.

Live mode is the default. It uses tiny or short descriptions during quick play and skips low-priority details. `schedule_speech(..., mode="replay")` is available in Python for more detail when timing permits.

### Media

- `commentary.wav` is a silent-padded timeline matching the video duration.
- `commentary.mp3` is the synchronized compressed narration track.
- `out.mp4` is the annotated video with scheduled text overlays.
- `out_described.mp4` combines the annotated video and synchronized narration.

Speech uses a Windows system voice when available, with `pyttsx3` as a portable fallback. FFmpeg handles MP3 conversion and video muxing. Speech failure does not remove TXT, SRT, or video outputs.

## Description style

The system prioritizes location, movement, direction, and result:

```text
Player 1 serves.
Wide serve.
Player 2 crosscourt.
Player 1 down the line.
Player 2 deep.
Player 1 misses.
Point Player 2.
```

It avoids hype, tactical speculation, forehand/backhand guesses, and narration for every bounce or small movement.

## Tests

Tests do not require model weights, video inference, or a GPU:

```bash
python -m unittest discover -s tests -v
```

They cover location language, shot direction, movement, templates, speech duration, scheduling skips, SRT timestamps, path helpers, and synthetic homography.

## Limitations

- Ball occlusion, blur, and leaving the frame remain difficult.
- Shot direction and point winner inference are approximate.
- Forehand and backhand are unreliable without pose estimation, so they are not guessed.
- Homography depends on visible, correctly identified court keypoints.
- Rule-based moment extraction may miss contacts or infer an uncertain point ending.
- Short synchronized descriptions are prioritized over complete analysis.
- Low-priority moments are intentionally silent during fast rallies.
