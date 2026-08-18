import yaml
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

OUT_DIR = Path("data")

def load_dataset() -> pd.DataFrame:
    url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    df = pd.read_csv(url)
    # The dataset has 'medv' as the target column, we rename it to 'label'
    df.rename(columns={"medv": "label"}, inplace=True)
    return df

def main():
    params = yaml.safe_load(open("params.yaml"))["prepare"]
    OUT_DIR.mkdir(exist_ok=True)

    df = load_dataset()
    train_df, test_df = train_test_split(
        df,
        test_size=params["test_size"],
        random_state=params["random_state"]
    )

    train_df.to_csv(OUT_DIR / "train.csv", index=False)
    test_df.to_csv(OUT_DIR / "test.csv", index=False)
    print(f"Wrote {len(train_df)} train rows, {len(test_df)} test rows")

if __name__ == "__main__":
    main()
