from pathlib import Path
import pickle
import streamlit as st

st.set_page_config(page_title="SMS Spam Classifier", page_icon="📩")


@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "spam_model.pkl"

    with open(model_path, "rb") as file:
        return pickle.load(file)


model = load_model()

st.title("SMS Spam Classifier")
st.write("Enter a message to predict whether it is spam or legitimate.")

message = st.text_area("Message", height=150)

if st.button("Check message"):
    if not message.strip():
        st.warning("Enter a message first.")
    else:
        prediction = model.predict([message])[0]

        if prediction == "spam":
            st.error("Prediction: Spam")
        else:
            st.success("Prediction: Not spam")

st.caption(
    "Learning project using TF-IDF and Multinomial Naive Bayes. "
    "Predictions can be incorrect."
)