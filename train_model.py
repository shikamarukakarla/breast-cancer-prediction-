from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "dataset"
MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

def main():
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data
    y = dataset.target
    df = X.copy()
    df["target"] = y
    df["target_label"] = y.map({0: "malignant", 1: "benign"})
    df.to_csv(DATA_DIR / "breast_cancer_wisconsin.csv", index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    candidates = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=5000, random_state=42))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced"
        )
    }

    comparison = {}
    fitted = {}
    for name, estimator in candidates.items():
        estimator.fit(X_train, y_train)
        predictions = estimator.predict(X_test)
        comparison[name] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1_score": float(f1_score(y_test, predictions, zero_division=0))
        }
        fitted[name] = estimator
        print(f"\n{name}")
        print(classification_report(y_test, predictions, target_names=dataset.target_names))
        print("Confusion matrix:\n", confusion_matrix(y_test, predictions))

    selected_name = max(comparison, key=lambda name: comparison[name]["f1_score"])
    joblib.dump(fitted[selected_name], MODEL_DIR / "best_model.joblib")

    metrics = {
        "selected_model": selected_name,
        "comparison": comparison,
        "test_size": int(len(y_test)),
        "random_state": 42,
        "target_names": list(dataset.target_names),
        "features": list(X.columns)
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nSelected model: {selected_name}")
    print(f"Saved model to: {MODEL_DIR / 'best_model.joblib'}")
    print(f"Saved dataset to: {DATA_DIR / 'breast_cancer_wisconsin.csv'}")

if __name__ == "__main__":
    main()
