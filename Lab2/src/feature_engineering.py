"""
Stage 3: Feature Engineering
------------------------------
Reads the processed CSV, scales numeric features with StandardScaler,
splits into train/test sets, and writes both to data/features/.

Input:
    data/processed/data.csv
Output:
    data/features/train.csv
    data/features/test.csv
"""

import os
import yaml
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_params(path="params.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_processed_data(path="data/processed/data.csv"):
    df = pd.read_csv(path)
    print(f"[feature_engineering] Loaded processed data (shape={df.shape})")
    return df


def build_features(df, test_size, random_state):
    X = df.drop(columns=["target"])
    y = df["target"]

    # No stratify for regression
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns
    )

    train_df = X_train_scaled.copy()
    train_df["target"] = y_train.reset_index(drop=True)

    test_df = X_test_scaled.copy()
    test_df["target"] = y_test.reset_index(drop=True)

    return train_df, test_df, scaler


def save_features(train_df, test_df, scaler, out_dir="data/features"):
    os.makedirs(out_dir, exist_ok=True)

    train_df.to_csv(f"{out_dir}/train.csv", index=False)
    test_df.to_csv(f"{out_dir}/test.csv", index=False)

    joblib.dump(scaler, f"{out_dir}/scaler.pkl")

    print(f"[feature_engineering] Saved train -> {out_dir}/train.csv")
    print(f"[feature_engineering] Saved test -> {out_dir}/test.csv")
    print(f"[feature_engineering] Saved scaler -> {out_dir}/scaler.pkl")


def main():
    params = load_params()["feature_engineering"]

    df = load_processed_data()

    train_df, test_df, scaler = build_features(
        df,
        test_size=params["test_size"],
        random_state=params["random_state"]
    )

    save_features(train_df, test_df, scaler)


if __name__ == "__main__":
    main()