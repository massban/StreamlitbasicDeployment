import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

st.title("KNN Weather Classification")

X = np.array([[50,70],[25,80],[27,60],[31,65],[23,85],[20,75]])
y = np.array([0,1,0,0,1,1])
labels = {0:"Sunny", 1:"Rainy"}

temp = st.slider("Temperature", 10, 60, 26)
hum = st.slider("Humidity", 50, 95, 78)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)
pred = model.predict([[temp, hum]])[0]

st.write(f"Predicted Weather: **{labels[pred]}**")

fig, ax = plt.subplots()
ax.scatter(X[y==0,0], X[y==0,1], label="Sunny")
ax.scatter(X[y==1,0], X[y==1,1], label="Rainy")
ax.scatter(temp, hum, marker="*", s=200, label="New Day")
ax.legend()
st.pyplot(fig)
