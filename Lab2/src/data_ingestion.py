"""
Stage 1: Data Ingestion
------------------------
Loads the Boston Housing dataset from OpenML and saves it as a raw CSV file.

Output:
    data/raw/data.csv
"""

import os
import pandas as pd
from sklearn.datasets import fetch_openml


def load_data() -> pd.DataFrame:
    """Load the Boston Housing dataset into a DataFrame."""
    bunch = fetch_openml(
        name="boston",
        version=1,
        as_frame=True
    )

    df = bunch.data
    df["target"] = bunch.target  # Add target column
    return df


def save_raw_data(df: pd.DataFrame, out_dir: str = "data/raw") -> None:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "data.csv")
    df.to_csv(out_path, index=False)
    print(f"[data_ingestion] Saved raw data -> {out_path} (shape={df.shape})")


def main():
    df = load_data()
    save_raw_data(df)


if __name__ == "__main__":
    main()