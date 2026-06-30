import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bounce_detection import BOUNCE_THRESHOLD, MODEL_PATH, trajectory_features  # noqa: E402

DATASET = ROOT / "training/ball/dataset"

def load_samples():
    data = pd.concat([
        pd.read_csv(DATASET / "labels_train.csv"),
        pd.read_csv(DATASET / "labels_val.csv"),
    ], ignore_index=True)
    data["clip"] = data["path1"].str.rsplit("/", n=1).str[0]
    data["frame"] = data["path1"].str.extract(r"/(\d+)\.jpg$")[0].astype(int)
    data["game"] = data["path1"].str.extract(r"images/game(\d+)/")[0].astype(int)
    features, labels, games = [], [], []
    data = data.sort_values(["clip", "frame"]).drop_duplicates(["clip", "frame"])
    for _, group in data.groupby("clip"):
        group = group.set_index("frame")
        timeline = np.arange(group.index.min(), group.index.max() + 1)
        expanded = group.reindex(timeline)
        clip_features = trajectory_features(
            expanded["x-coordinate"].to_numpy(), expanded["y-coordinate"].to_numpy()
        )
        for frame, row in group.iterrows():
            features.append(clip_features[frame - timeline[0]])
            labels.append(int(row["status"] == 2))
            games.append(int(row["game"]))
    return np.asarray(features), np.asarray(labels), np.asarray(games)

def new_model():
    return CatBoostClassifier(
        iterations=350, depth=7, learning_rate=0.06, loss_function="Logloss",
        class_weights=[1, 5], random_seed=42, verbose=False, allow_writing_files=False,
    )

def main():
    features, labels, games = load_samples()
    train, test = games <= 6, games >= 8
    evaluation = new_model()
    evaluation.fit(features[train], labels[train])
    predicted = evaluation.predict_proba(features[test])[:, 1] >= BOUNCE_THRESHOLD
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels[test], predicted, average="binary", zero_division=0
    )
    print(f"accuracy={accuracy_score(labels[test], predicted):.6f}")
    print(f"precision={precision:.6f} recall={recall:.6f} f1={f1:.6f}")
    final = new_model()
    final.fit(features, labels)
    final.save_model(str(MODEL_PATH))
    print(f"saved={MODEL_PATH}")

if __name__ == "__main__":
    main()