from pathlib import Path
import argparse
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset_player"
DATA_YAML = DATASET_DIR / "data.yaml"
RUNS_DIR = ROOT / "runs" / "detect"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--name", default="player_detector")

    return parser.parse_args()


def resolve_model_path(model_arg):
    model_path = Path(model_arg)

    if model_path.exists():
        return str(model_path)

    project_model_path = ROOT / model_arg

    if project_model_path.exists():
        return str(project_model_path)

    return model_arg


def main():
    args = parse_args()

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Cannot find data.yaml: {DATA_YAML}")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = resolve_model_path(args.model)

    print("Project root:")
    print(ROOT)

    print("Dataset YAML:")
    print(DATA_YAML)

    print("Training model:")
    print(model_path)

    model = YOLO(model_path)

    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(RUNS_DIR),
        name=args.name,
        exist_ok=True
    )

    best_model_path = RUNS_DIR / args.name / "weights" / "best.pt"
    last_model_path = RUNS_DIR / args.name / "weights" / "last.pt"

    print("Training finished.")
    print("Best model:")
    print(best_model_path)

    print("Last model:")
    print(last_model_path)


if __name__ == "__main__":
    main()