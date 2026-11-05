# SMS Spam Classifier

A small machine-learning application that classifies SMS messages as
spam or legitimate (ham).

Built with Python, pandas, scikit-learn, and Streamlit.

## How it works

1. Load labelled messages from the UCI SMS Spam Collection.
2. Remove duplicate messages.
3. Split the data into 80% training and 20% testing, preserving class proportions.
4. Convert messages into numerical features using TF-IDF.
5. Train a Multinomial Naive Bayes classifier.
6. Evaluate predictions using precision, recall, F1, and a confusion matrix.
7. Use Streamlit to classify messages entered by the user.

TF-IDF is fitted only on the training data. A scikit-learn Pipeline
keeps preprocessing and classification together.

## Dataset

Source: UCI SMS Spam Collection
https://archive.ics.uci.edu/dataset/228/sms+spam+collection

The downloaded file contained 5,572 messages.
After removing duplicate messages, 5,169 remained:
- 4,516 legitimate messages
- 653 spam messages

The split used 4,135 training messages and 1,034 test messages,
with random_state=42 and stratification by label.

## Experiment results

| Metric | alpha=1.0 | alpha=0.1 |
|---|---:|---:|
| Accuracy | 95.3% | 98.3% |
| Spam precision | 100% | 97.5% |
| Spam recall | 62.6% | 88.5% |
| Spam F1 | 77.0% | 92.8% |
| Missed spam messages | 49 | 15 |
| Legitimate messages flagged as spam | 0 | 3 |

I selected alpha=0.1 for the demo because it caught more spam,
with a small increase in false alarms.

Both settings were compared on the same held-out split.
Because this comparison informed the parameter choice, the results
are exploratory. A fresh evaluation set would be needed for an
independent assessment of the selected model.

## Run locally

Requires Python and pip.

Create a virtual environment on Windows:

    py -m venv .venv

Install dependencies:

    .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Download and extract the dataset from:
https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip

Place the extracted SMSSpamCollection file inside a folder named data.

Train and save the model:

    .\.venv\Scripts\python.exe train.py

Launch the interface:

    .\.venv\Scripts\python.exe -m streamlit run app.py

Training creates spam_model.pkl. Restart Streamlit after retraining
to reload the cached model.

## Limitations

- The model can miss spam and incorrectly flag legitimate messages.
- Results on this dataset may not reflect performance on modern SMS.
- It learns statistical text patterns; it does not verify links or senders.
- The interface is a learning demo, not a production spam filter.
- Only load pickle model files you trust.

## Development

Built as a learning project with AI-assisted guidance.
The dataset is provided by its original contributors through UCI.