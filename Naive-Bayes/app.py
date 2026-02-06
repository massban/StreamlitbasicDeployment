import streamlit as st
import joblib

model = joblib.load("model.joblib")
tfidf = joblib.load("tfidf.joblib")

def predict_sentiment(review):
    review_tfidf = tfidf.transform([review])
    prediction = model.predict(review_tfidf)
    return "Positive" if prediction[0] == 1 else "Negative"

st.title("Movie Review Sentiment Analysis")

review = st.text_area("Enter a movie review")

if st.button("Predict"):
    if review.strip():
        st.success(predict_sentiment(review))
    else:
        st.warning("Please enter a review")
