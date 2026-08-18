import sys
import json
import yaml
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def main():
    params = yaml.safe_load(open("params.yaml"))["evaluate"]

    clf = joblib.load("model/model.joblib")
    test_df = pd.read_csv("data/test.csv")
    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"]

    preds = clf.predict(X_test)

    metrics = {
        "mse": float(mean_squared_error(y_test, preds)),
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2_score": float(r2_score(y_test, preds))
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))

    if metrics["r2_score"] < params["min_r2_score"]:
        print(
            f"FAIL: R2 Score {metrics['r2_score']:.4f} "
            f"is below gate {params['min_r2_score']}"
        )
        sys.exit(1)

    print("PASS: model cleared the quality gate")

if __name__ == "__main__":
    main()
