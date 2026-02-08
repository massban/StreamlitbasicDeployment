import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

st.title("Spam Email Detection")

emails = [
    "Win a free iPhone",
    "Meeting tomorrow",
    "Claim your prize",
    "Project discussion",
    "Limited offer"
]
labels = [1,0,1,0,1]

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(emails)

model = LinearSVC()
model.fit(X, labels)

msg = st.text_area("Enter Email")
if st.button("Check"):
    pred = model.predict(vectorizer.transform([msg]))[0]
    st.write("Spam Email" if pred==1 else "Not Spam Email")
