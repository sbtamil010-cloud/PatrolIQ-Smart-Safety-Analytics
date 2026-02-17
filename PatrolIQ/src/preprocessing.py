# src/preprocessing.py

import pandas as pd


def load_and_sample_data(filepath, sample_size=500000):
    """
    Load Chicago crime dataset and sample recent records.
    """
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values("Date", ascending=False)
    df = df.head(sample_size)
    return df


def clean_data(df):
    """
    Clean missing values and remove invalid coordinates.
    """
    df = df.dropna(subset=['Latitude', 'Longitude'])
    df.fillna("Unknown", inplace=True)
    return df


def extract_temporal_features(df):
    """
    Create temporal features from Date column.
    """
    df['Hour'] = df['Date'].dt.hour
    df['Day_of_Week'] = df['Date'].dt.day_name()
    df['Month'] = df['Date'].dt.month

    df['Is_Weekend'] = df['Day_of_Week'].isin(['Saturday', 'Sunday'])

    return df
