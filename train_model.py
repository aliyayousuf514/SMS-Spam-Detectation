
from pathlib import Path
import zipfile

import joblib
import pandas as pd
import requests

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from preprocessing import clean_text


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

DATA_URL = (
    "https://archive.ics.uci.edu/static/public/228/"
    "sms+spam+collection.zip"
)
ZIP_PATH = DATA_DIR / "sms_spam.zip"
DATASET_PATH = DATA_DIR / "SMSSpamCollection"


def download_dataset():
    if DATASET_PATH.exists():
        return

    response = requests.get(DATA_URL, timeout=60)
    response.raise_for_status()
    ZIP_PATH.write_bytes(response.content)

    with zipfile.ZipFile(ZIP_PATH, "r") as archive:
        matches = [
            name for name in archive.namelist()
            if name.endswith("SMSSpamCollection")
        ]

        if not matches:
            raise FileNotFoundError(
                "SMSSpamCollection file was not found in the ZIP."
            )

        with archive.open(matches[0]) as source:
            DATASET_PATH.write_bytes(source.read())


def load_dataset():
    download_dataset()

    df = pd.read_csv(
        DATASET_PATH,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8",
    )

    df = df.dropna(subset=["label", "message"])
    df["label"] = df["label"].str.lower().str.strip()
    df["message"] = df["message"].apply(clean_text)

    df = df[df["label"].isin(["ham", "spam"])]
    df = df[df["message"] != ""]

    return df


def train_models():
    df = load_dataset()

    X = df["message"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    algorithms = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        ),
        "SVM": LinearSVC(class_weight="balanced"),
    }

    results = []
    trained_models = {}

    for name, algorithm in algorithms.items():
        pipeline = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            ("classifier", algorithm),
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        results.append({
            "Model": name,
            "Accuracy": round(
                accuracy_score(y_test, predictions), 4
            ),
            "Precision": round(
                precision_score(
                    y_test, predictions, pos_label="spam",
                    zero_division=0,
                ), 4
            ),
            "Recall": round(
                recall_score(
                    y_test, predictions, pos_label="spam",
                    zero_division=0,
                ), 4
            ),
            "F1 Score": round(
                f1_score(
                    y_test, predictions, pos_label="spam",
                    zero_division=0,
                ), 4
            ),
        })

        trained_models[name] = pipeline

    results_df = pd.DataFrame(results)
    best_name = results_df.sort_values(
        "F1 Score", ascending=False
    ).iloc[0]["Model"]

    joblib.dump(
        trained_models[best_name],
        MODEL_DIR / "spam_model.joblib",
    )

    joblib.dump(
        results_df,
        MODEL_DIR / "model_results.joblib",
    )

    print("\nModel Comparison:")
    print(results_df.to_string(index=False))
    print(f"\nSelected model: {best_name}")
    print(f"Dataset messages: {len(df)}")
    print("Saved model to models/spam_model.joblib")

    return results_df


if __name__ == "__main__":
    train_models()
