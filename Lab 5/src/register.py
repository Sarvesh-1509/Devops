import os
import json
from huggingface_hub import HfApi, create_repo

REPO_ID = os.environ["HF_REPO_ID"]     
HF_TOKEN = os.environ["HF_TOKEN"]

def build_model_card(metrics: dict) -> str:
    lines = [
        "---",
        "tags: [sklearn, random-forest, boston-housing, regression, mlops-pipeline]",
        "---",
        "# Boston Housing Regression Model",
        "",
        "Trained automatically via GitHub Actions CI/CD. ",
        "",
        "## Metrics",
        "",
    ]
    for k, v in metrics.items():
        lines.append(f"- **{k}**: {v:.4f}")
    return "\n".join(lines)

def main():
    with open("metrics.json") as f:
        metrics = json.load(f)

    api = HfApi(token=HF_TOKEN)
    create_repo(REPO_ID, token=HF_TOKEN, exist_ok=True, repo_type="model")

    with open("model/README.md", "w") as f:
        f.write(build_model_card(metrics))

    for path in ["model/model.joblib", "model/features.json", "model/README.md"]:
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=os.path.basename(path),
            repo_id=REPO_ID,
            token=HF_TOKEN,
            repo_type="model",
        )

    print(f"Model pushed to https://huggingface.co/{REPO_ID}")

if __name__ == "__main__":
    main()
