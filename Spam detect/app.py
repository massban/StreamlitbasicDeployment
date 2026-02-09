import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

st.set_page_config(page_title="Spam Detection", layout="centered")
st.title("📧 Spam Email Detection")

emails = [
    "Win a free iPhone",
    "Meeting tomorrow",
    "Claim your prize now",
    "Project discussion",
    "Limited offer today"
]
labels = [1,0,1,0,1]

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(emails)

model = LinearSVC()
model.fit(X, labels)

msg = st.text_area("Enter Email Message")
if st.button("Check"):
    pred = model.predict(vectorizer.transform([msg]))[0]
    st.success("Spam Email" if pred else "Not Spam Email")
