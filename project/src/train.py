import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ==============================
# 1. Load dataset
# ==============================

data = pd.read_csv("data/breast_cancer.csv")

X = data.drop("target", axis=1)
y = data["target"]


# ==============================
# 2. Train-test split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==============================
# 3. Define models
# ==============================

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC())
    ])
}


# ==============================
# 4. MLflow experiment
# ==============================

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Breast Cancer Classification")


# Store results
results = {}


# ==============================
# 5. Train models
# ==============================

for name, model in models.items():

    print(f"\nTraining {name}...")

    with mlflow.start_run(run_name=name):

        # Train
        model.fit(X_train, y_train)

        # Predict
        predictions = model.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        # Store results
        results[name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

        # ==============================
        # MLflow parameters
        # ==============================

        mlflow.log_param("model_name", name)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        # ==============================
        # MLflow metrics
        # ==============================

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # ==============================
        # Log model artifact
        # ==============================

        mlflow.sklearn.log_model(
            model,
            "model"
        )

        # Print results
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")


# ==============================
# 6. Compare models
# ==============================

print("\n========== MODEL COMPARISON ==========")

for name, metrics in results.items():

    print(
        f"{name}: "
        f"Accuracy={metrics['accuracy']:.4f}, "
        f"F1={metrics['f1_score']:.4f}"
    )


# ==============================
# 7. Select best model
# ==============================

best_model_name = max(
    results,
    key=lambda name: results[name]["f1_score"]
)

best_model = models[best_model_name]

print(f"\nBest Model: {best_model_name}")


# ==============================
# 8. Save best model
# ==============================

joblib.dump(
    best_model,
    "models/best_model.pkl"
)

print("Best model saved to models/best_model.pkl")
