import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from audio_description import (
    classify_ball_location,
    classify_movement,
    classify_player_location,
    classify_shot_direction,
    make_descriptions,
)
from homography import transform_point
from pipeline_utils import ensure_output_dirs, organized_path, parse_number
from speech_scheduler import (
    choose_text,
    estimate_speech_seconds,
    format_srt_timestamp,
    schedule_speech,
)


class UtilityTests(unittest.TestCase):
    def test_numeric_parsing(self):
        self.assertEqual(parse_number("3.25"), 3.25)
        self.assertIsNone(parse_number("nan"))
        self.assertIsNone(parse_number("bad"))

    def test_output_path(self):
        with tempfile.TemporaryDirectory() as directory:
            ensure_output_dirs(directory)
            self.assertEqual(
                organized_path(directory, "moments"),
                str(Path(directory) / "narration" / "moments.csv"),
            )

    def test_synthetic_homography(self):
        matrix = np.array([[2.0, 0.0, 10.0], [0.0, 3.0, -4.0], [0.0, 0.0, 1.0]])
        self.assertEqual(transform_point(matrix, 5.0, 6.0), (20.0, 14.0))


class DescriptionLogicTests(unittest.TestCase):
    def test_locations(self):
        self.assertEqual(classify_player_location(0.5, 0.9), "center")
        self.assertEqual(classify_player_location(0.2, 1.05), "behind baseline")
        self.assertEqual(classify_ball_location(-0.1, 0.4), "out wide")
        self.assertEqual(classify_ball_location(0.5, 0.02), "baseline")

    def test_shot_direction(self):
        self.assertEqual(classify_shot_direction((0.2, 0.9), (0.7, 0.25)), "crosscourt")
        self.assertEqual(classify_shot_direction((0.2, 0.9), (0.25, 0.25)), "down the line")

    def test_recovery_movement(self):
        track = [
            {"frame": 2, "x": 0.2, "y": 0.9, "confidence": 0.9},
            {"frame": 10, "x": 0.46, "y": 0.9, "confidence": 0.9},
        ]
        movement, confidence = classify_movement(track, 10)
        self.assertEqual(movement, "recovers to center")
        self.assertGreater(confidence, 0.7)

    def test_long_short_tiny_templates(self):
        long, short, tiny = make_descriptions(
            "rally_shot", "player_1", movement="", direction="down the line",
        )
        self.assertEqual(long, "Player 1 hits down the line.")
        self.assertEqual(short, "Player 1 down the line.")
        self.assertEqual(tiny, "Down The Line.")


class SchedulerTests(unittest.TestCase):
    FIELDS = [
        "moment_id", "start_timestamp", "end_timestamp", "moment_type",
        "player_id", "importance", "player_location", "player_movement",
        "ball_location", "ball_direction", "result", "confidence",
        "description_long", "description_short", "description_tiny",
    ]

    def test_speech_duration_and_srt(self):
        self.assertAlmostEqual(estimate_speech_seconds("Player 1 serves."), 1.25)
        self.assertEqual(format_srt_timestamp(65.432), "00:01:05,432")

    def test_fast_rally_uses_tiny_text(self):
        moment = {
            "importance": 2,
            "description_long": "Player 1 moves right and hits down the line.",
            "description_short": "Player 1 down the line.",
            "description_tiny": "Down the line.",
        }
        text, source, _ = choose_text(moment, 1.3, "live")
        self.assertEqual(text, "Down the line.")
        self.assertEqual(source, "description_tiny")

    def test_scheduler_skips_low_priority_and_keeps_point(self):
        rows = [
            {
                "moment_id": "M0001", "start_timestamp": 1.0, "end_timestamp": 1.2,
                "moment_type": "player_movement", "importance": 3, "confidence": 0.8,
                "description_long": "Player 1 moves toward the deuce side.",
                "description_short": "Player 1 moves.", "description_tiny": "",
            },
            {
                "moment_id": "M0002", "start_timestamp": 2.0, "end_timestamp": 2.2,
                "moment_type": "point_result", "importance": 1, "confidence": 0.8,
                "description_long": "Point to Player 2.",
                "description_short": "Point Player 2.", "description_tiny": "Point Player 2.",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            ensure_output_dirs(directory)
            path = Path(organized_path(directory, "moments"))
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            lines = schedule_speech(directory, timeline_end=5.0, video_path="missing.mp4")
            self.assertTrue(Path(organized_path(directory, "scheduled_lines")).exists())

        self.assertEqual(lines[0]["was_skipped"], "True")
        self.assertEqual(lines[1]["was_skipped"], "False")
        self.assertEqual(lines[1]["spoken_text"], "Point Player 2.")


if __name__ == "__main__":
    unittest.main()
