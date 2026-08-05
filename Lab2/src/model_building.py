"""
Stage 4: Model Building
--------------------------
Trains a Linear Regression model on the engineered training features
and serializes the fitted model.

Input:
    data/features/train.csv
Output:
    model.pkl
"""

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression


def load_train_data(path="data/features/train.csv"):
    df = pd.read_csv(path)
    print(f"[model_building] Loaded training data (shape={df.shape})")
    return df


def train_model(df):
    X_train = df.drop(columns=["target"])
    y_train = df["target"]

    model = LinearRegression()
    model.fit(X_train, y_train)

    print("[model_building] Model training complete")
    return model


def save_model(model, path="model.pkl"):
    joblib.dump(model, path)
    print(f"[model_building] Saved model -> {path}")


def main():
    df = load_train_data()
    model = train_model(df)
    save_model(model)


if __name__ == "__main__":
    main()