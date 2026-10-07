# AI-Powered Email Spam Classifier
# Module 6 - Final AI & ML Project

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def train_model(data):
    """
    Train a Logistic Regression model to classify messages as
    spam or ham (not spam).
    """

    X = data["message"]
    y = data["label"]

    # Convert text into numerical features
    vectorizer = TfidfVectorizer()

    X_vectorized = vectorizer.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Create and train the ML model
    model = LogisticRegression(random_state=42)

    model.fit(X_train, y_train)

    # Test the model
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    return model, vectorizer, accuracy