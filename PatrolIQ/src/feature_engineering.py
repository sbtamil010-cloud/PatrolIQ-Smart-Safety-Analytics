# src/feature_engineering.py

import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def add_season(df):
    def get_season(month):
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Fall"

    df['Season'] = df['Month'].apply(get_season)
    return df


def add_crime_severity(df):
    severity_map = {
        'HOMICIDE': 5,
        'CRIM SEXUAL ASSAULT': 4,
        'ROBBERY': 3,
        'BATTERY': 2,
        'THEFT': 1
    }

    df['Crime_Severity'] = df['Primary Type'].map(severity_map)
    df['Crime_Severity'].fillna(1, inplace=True)

    return df


def encode_crime_type(df):
    le = LabelEncoder()
    df['Crime_Type_Encoded'] = le.fit_transform(df['Primary Type'])
    return df, le


def scale_geographic_features(df):
    scaler = StandardScaler()
    geo_scaled = scaler.fit_transform(df[['Latitude', 'Longitude']])
    return geo_scaled, scaler
