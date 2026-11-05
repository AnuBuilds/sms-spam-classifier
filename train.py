from pathlib import Path
import pandas as pd

# Locate the dataset relative to this Python file.
data_path = Path(__file__).parent / "data" / "SMSSpamCollection"

# Each row contains a label and a message, separated by a tab.
data = pd.read_csv(
    data_path,
    sep="\t",
    header=None,
    names=["label", "message"],
    keep_default_na=False
)

print("First five messages:")
print(data.head())

print("\nOriginal number of messages:", len(data))

# Remove repeated messages before splitting into training and test sets.
data = data.drop_duplicates(subset=["message"])

print("Messages after removing duplicates:", len(data))

print("\nNumber of messages in each category:")
print(data["label"].value_counts())

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# Use 80% for training and keep 20% unseen for testing.
# Stratify keeps similar proportions of spam in both groups.
X_train, X_test, y_train, y_test = train_test_split(
    data["message"],
    data["label"],
    test_size=0.2,
    random_state=42,
    stratify=data["label"]
)

# First convert text into numerical features, then classify it.
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB(alpha=0.1))
])

# The vocabulary and classifier learn only from training messages.
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
    zero_division=0
))

print("Confusion matrix:")
print("Rows = actual; columns = predicted")
print("Order: ham, spam")
print(confusion_matrix(y_test, predictions, labels=["ham", "spam"]))

# Save both the text processor and classifier together.
model_path = Path(__file__).parent / "spam_model.pkl"

with open(model_path, "wb") as file:
    pickle.dump(model, file)

print("\nModel saved to:", model_path)