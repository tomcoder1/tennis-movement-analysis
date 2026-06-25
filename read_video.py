from collections import deque
from dataclasses import dataclass
from typing import Iterator, Optional, List

import cv2
import numpy as np

@dataclass
class FramePacket:
    frame_id: int
    timestamp: float
    frame_bgr: np.ndarray
    tracknet_window_bgr: Optional[List[np.ndarray]]
    fps: float

def read_video(video_path: str) -> Iterator[FramePacket]:
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        raise RuntimeError("Could not read FPS from video.")

    previous_frames = deque(maxlen=2)
    frame_id = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        timestamp = frame_id / fps

        tracknet_window = None

        if len(previous_frames) == 2:
            tracknet_window = [
                frame_bgr,
                previous_frames[-1],
                previous_frames[-2],
            ]

        yield FramePacket(
            frame_id=frame_id,
            timestamp=timestamp,
            frame_bgr=frame_bgr,
            tracknet_window_bgr=tracknet_window,
            fps=fps,
        )

        previous_frames.append(frame_bgr)
        frame_id += 1

    cap.release()