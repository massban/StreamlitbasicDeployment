import streamlit as st
import pandas as pd
import numpy as np
import math

st.title("Decision Tree Classifier (ID3)")

data = pd.DataFrame({
    "Outlook": [
        "Sunny","Sunny","Overcast","Rain","Rain","Rain",
        "Overcast","Sunny","Sunny","Rain","Sunny","Overcast",
        "Overcast","Rain"
    ],
    "Humidity": [
        "High","High","High","High","Normal","Normal",
        "Normal","High","Normal","High","Normal","High",
        "Normal","High"
    ],
    "PlayTennis": [
        "No","No","Yes","Yes","No","Yes",
        "Yes","No","Yes","Yes","Yes","Yes",
        "Yes","No"
    ]
})

def entropy(col):
    values, counts = np.unique(col, return_counts=True)
    ent = 0
    for c in counts:
        p = c / len(col)
        ent -= p * math.log2(p)
    return ent

def information_gain(df, attribute, target):
    total_entropy = entropy(df[target])
    values, counts = np.unique(df[attribute], return_counts=True)
    weighted_entropy = 0
    for i in range(len(values)):
        subset = df[df[attribute] == values[i]]
        weighted_entropy += (counts[i] / len(df)) * entropy(subset[target])
    return total_entropy - weighted_entropy

def id3(df, target, attributes):
    if len(np.unique(df[target])) == 1:
        return df[target].iloc[0]
    if len(attributes) == 0:
        return df[target].mode()[0]

    gains = [information_gain(df, attr, target) for attr in attributes]
    best_attr = attributes[np.argmax(gains)]

    tree = {best_attr: {}}
    for value in np.unique(df[best_attr]):
        subset = df[df[best_attr] == value]
        remaining_attrs = [attr for attr in attributes if attr != best_attr]
        tree[best_attr][value] = id3(subset, target, remaining_attrs)

    return tree

def predict(tree, sample):
    if not isinstance(tree, dict):
        return tree
    attr = next(iter(tree))
    value = sample.get(attr)
    if value in tree[attr]:
        return predict(tree[attr][value], sample)
    else:
        return "Unknown"

attributes = ["Outlook", "Humidity"]

if st.button("Generate Decision Tree"):
    decision_tree = id3(data, "PlayTennis", attributes)
    st.subheader("Decision Tree")
    st.json(decision_tree)
    st.session_state["tree"] = decision_tree

if "tree" in st.session_state:
    st.subheader("Prediction")
    outlook = st.selectbox("Outlook", data["Outlook"].unique())
    humidity = st.selectbox("Humidity", data["Humidity"].unique())

    if st.button("Predict"):
        sample = {"Outlook": outlook, "Humidity": humidity}
        result = predict(st.session_state["tree"], sample)
        st.write(f"PlayTennis: **{result}**")
