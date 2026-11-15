from pathlib import Path
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent


def load_data(data_path):
    """Load messages and remove duplicates."""
    data = pd.read_csv(
        data_path,
        sep="\t",
        header=None,
        names=["label", "message"],
        keep_default_na=False,
    )

    print("First five messages:")
    print(data.head())
    print("\nOriginal number of messages:", len(data))

    data = data.drop_duplicates(subset=["message"])

    print("Messages after removing duplicates:", len(data))
    print("\nNumber of messages in each category:")
    print(data["label"].value_counts())

    return data


def create_model():
    """Create the text-processing and classification pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", MultinomialNB(alpha=0.1)),
    ])


def main(data_path=None, model_path=None):
    """Train, evaluate, and save the model."""
    if data_path is None:
        data_path = BASE_DIR / "data" / "SMSSpamCollection"

    if model_path is None:
        model_path = BASE_DIR / "spam_model.pkl"

    data = load_data(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        data["message"],
        data["label"],
        test_size=0.2,
        random_state=42,
        stratify=data["label"],
    )

    model = create_model()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\nTraining messages:", len(X_train))
    print("Testing messages:", len(X_test))

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=["ham", "spam"],
        digits=3,
        zero_division=0,
    ))

    print("Confusion matrix:")
    print("Rows = actual; columns = predicted")
    print("Order: ham, spam")
    print(confusion_matrix(
        y_test, predictions, labels=["ham", "spam"]
    ))

    with open(model_path, "wb") as file:
        pickle.dump(model, file)

    print("\nModel saved to:", model_path)
    return model


if __name__ == "__main__":
    main()