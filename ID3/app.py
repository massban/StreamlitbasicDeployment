import streamlit as st
import pandas as pd
import numpy as np
import math

st.title("ID3 Decision Tree")

def entropy(col):
    _, counts = np.unique(col, return_counts=True)
    return -sum((c/len(col))*math.log2(c/len(col)) for c in counts)

def info_gain(df, attr, target):
    total = entropy(df[target])
    values = df[attr].unique()
    return total - sum(
        (len(df[df[attr]==v])/len(df))*entropy(df[df[attr]==v][target])
        for v in values
    )

def id3(df, target, attrs):
    if len(df[target].unique()) == 1:
        return df[target].iloc[0]
    if not attrs:
        return df[target].mode()[0]
    best = max(attrs, key=lambda x: info_gain(df, x, target))
    tree = {best:{}}
    for v in df[best].unique():
        tree[best][v] = id3(df[df[best]==v], target, [a for a in attrs if a!=best])
    return tree

data = {
    "outlook":["sunny","sunny","overcast","rain","rain"],
    "humidity":["high","normal","high","normal","high"],
    "playtennis":["no","yes","yes","yes","no"]
}
df = pd.DataFrame(data)

if st.button("Train Model"):
    tree = id3(df, "playtennis", ["outlook","humidity"])
    st.json(tree)
