# AI-Powered Email Spam Classifier
# Module 6 - Final AI & ML Project

import os
import pandas as pd
import streamlit as st

from model import train_model


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="AI Email Spam Classifier",
    page_icon="📧",
    layout="centered"
)


# -----------------------------
# Title
# -----------------------------

st.title("📧 AI-Powered Email Spam Classifier")

st.write(
    "Enter an email or message below and the Machine Learning model "
    "will predict whether it is Spam or Not Spam."
)


# -----------------------------
# Load Dataset
# -----------------------------

DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "data",
    "spam_dataset.csv"
)

data = pd.read_csv(DATA_PATH)


# -----------------------------
# Train Model
# -----------------------------

model, vectorizer, accuracy = train_model(data)


# -----------------------------
# Message Input
# -----------------------------

st.subheader("✉️ Enter a Message")

message = st.text_area(
    "Type or paste your email/message here:",
    height=150,
    placeholder="Example: Congratulations! You have won a free prize..."
)


# -----------------------------
# Prediction
# -----------------------------

if st.button("🔍 Predict Message"):

    if message.strip() == "":
        st.warning("Please enter a message before predicting.")

    else:
        message_vectorized = vectorizer.transform([message])

        prediction = model.predict(message_vectorized)[0]

        probabilities = model.predict_proba(message_vectorized)[0]

        confidence = max(probabilities) * 100


        # -----------------------------
        # Display Result
        # -----------------------------

        st.subheader("📊 Prediction Result")

        if prediction == "spam":

            st.error("🚨 SPAM MESSAGE")

            st.write(
                f"Confidence: **{confidence:.2f}%**"
            )

        else:

            st.success("✅ NOT SPAM")

            st.write(
                f"Confidence: **{confidence:.2f}%**"
            )


# -----------------------------
# Model Information
# -----------------------------

st.subheader("🤖 Model Information")

st.write(
    "This application uses TF-IDF text vectorization and "
    "Logistic Regression for spam classification."
)

st.write(
    f"Model Accuracy: **{accuracy * 100:.2f}%**"
)


# -----------------------------
# Dataset Information
# -----------------------------

st.subheader("📚 Dataset Information")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Total Messages",
        len(data)
    )

with col2:
    st.metric(
        "Spam Messages",
        len(data[data["label"] == "spam"])
    )


# -----------------------------
# Footer
# -----------------------------

st.info(
    "This project was developed as part of Module 6 - Final AI & ML Project "
    "for the Codomax Digital Solutions internship."
)