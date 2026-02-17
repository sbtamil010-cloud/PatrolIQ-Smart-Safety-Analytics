# streamlit_app/pages/1_EDA.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("📊 Exploratory Data Analysis")

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/processed_crime.csv")

df = load_data()

st.subheader("Dataset Overview")
st.write(df.head())
st.write("Shape:", df.shape)

# Crime Type Distribution
st.subheader("Crime Type Distribution")
crime_counts = df['Primary Type'].value_counts().head(15)

fig, ax = plt.subplots(figsize=(10,5))
crime_counts.plot(kind='bar', ax=ax)
st.pyplot(fig)

# Hourly Crime Pattern
st.subheader("Crime Distribution by Hour")
hour_counts = df['Hour'].value_counts().sort_index()

fig2, ax2 = plt.subplots()
hour_counts.plot(ax=ax2)
st.pyplot(fig2)
