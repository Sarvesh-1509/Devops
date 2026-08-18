import yaml
import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

OUT_DIR = Path("model")

def main():
    params = yaml.safe_load(open("params.yaml"))["train"]
    OUT_DIR.mkdir(exist_ok=True)

    train_df = pd.read_csv("data/train.csv")
    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"]

    clf = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=params["random_state"],
    )
    clf.fit(X_train, y_train)

    joblib.dump(clf, OUT_DIR / "model.joblib")
    
    with open(OUT_DIR / "features.json", "w") as f:
        json.dump(list(X_train.columns), f)
        
    print(f"Trained RandomForestRegressor and saved to {OUT_DIR}/model.joblib")

if __name__ == "__main__":
    main()
