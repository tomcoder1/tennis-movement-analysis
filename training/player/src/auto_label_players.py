from pathlib import Path
import argparse
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset_player"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    return parser.parse_args()


def convert_box_to_yolo(box, image_width, image_height):
    x1, y1, x2, y2 = box

    x_center = ((x1 + x2) / 2) / image_width
    y_center = ((y1 + y2) / 2) / image_height
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height

    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def main():
    args = parse_args()

    image_dir = DATASET_DIR / "images" / args.split
    label_dir = DATASET_DIR / "labels" / args.split

    label_dir.mkdir(parents=True, exist_ok=True)

    if not image_dir.exists():
        raise FileNotFoundError(f"Image folder not found: {image_dir}")

    model = YOLO(args.model)

    image_paths = sorted(image_dir.glob("*.jpg"))

    if len(image_paths) == 0:
        raise RuntimeError(f"No jpg images found in {image_dir}")

    for image_path in image_paths:
        results = model.predict(
            source=str(image_path),
            classes=[0],
            conf=args.conf,
            verbose=False
        )

        result = results[0]
        image_height, image_width = result.orig_shape

        label_lines = []

        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                box_width = x2 - x1
                box_height = y2 - y1

                # Ignore tiny detections
                if box_height < image_height * 0.04:
                    continue

                # Ignore huge wrong detections
                if box_height > image_height * 0.80:
                    continue

                # Ignore very wide detections
                if box_width > image_width * 0.35:
                    continue

                line = convert_box_to_yolo(
                    [x1, y1, x2, y2],
                    image_width,
                    image_height
                )

                label_lines.append(line)

        label_path = label_dir / f"{image_path.stem}.txt"

        with open(label_path, "w", encoding="utf-8") as f:
            for line in label_lines:
                f.write(line + "\n")

    print("Done.")
    print(f"Auto labels saved to: {label_dir}")


if __name__ == "__main__":
    main()