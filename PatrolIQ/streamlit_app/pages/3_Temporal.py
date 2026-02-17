# streamlit_app/pages/3_Temporal.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("⏰ Temporal Crime Patterns")

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/processed_crime.csv")

df = load_data()

# Hourly Pattern
st.subheader("Hourly Crime Pattern")
hour_counts = df.groupby("Hour").size()

fig, ax = plt.subplots()
hour_counts.plot(ax=ax)
st.pyplot(fig)

# Weekend vs Weekday
st.subheader("Weekend vs Weekday Crimes")
weekend_counts = df.groupby("Is_Weekend").size()

fig2, ax2 = plt.subplots()
weekend_counts.plot(kind="bar", ax=ax2)
st.pyplot(fig2)
