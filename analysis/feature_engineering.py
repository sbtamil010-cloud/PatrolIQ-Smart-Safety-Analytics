# src/analysis/feature_engineering.py
"""
feature_engineering.py
------------------------------------
Generates model-ready features for the PatrolIQ project.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH_01 = BASE_DIR / "data" / "processed" / "sample_250000_rows_01.csv"
DATA_PATH_02 = BASE_DIR / "data" / "processed" / "sample_250000_rows_02.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "model_ready_data.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True, parents=True)


def assign_season(month):
    if month in [12, 1, 2]: return "Winter"
    elif month in [3, 4, 5]: return "Spring"
    elif month in [6, 7, 8]: return "Summer"
    return "Fall"


def create_crime_severity(type_):
    severity_map = {
        "HOMICIDE": 5, "CRIM SEXUAL ASSAULT": 5, "CRIMINAL SEXUAL ASSAULT": 5,
        "KIDNAPPING": 5, "HUMAN TRAFFICKING": 5,
        "ASSAULT": 4, "BATTERY": 4, "ROBBERY": 4, "ARSON": 4,
        "WEAPONS VIOLATION": 4, "INTIMIDATION": 4, "STALKING": 4,
        "OFFENSE INVOLVING CHILDREN": 4,
        "BURGLARY": 3, "MOTOR VEHICLE THEFT": 3, "CRIMINAL DAMAGE": 3,
        "CRIMINAL TRESPASS": 3, "INTERFERENCE WITH PUBLIC OFFICER": 3,
        "CONCEALED CARRY LICENSE VIOLATION": 3,
        "THEFT": 2, "DECEPTIVE PRACTICE": 2, "NARCOTICS": 2,
        "OTHER OFFENSE": 2, "OTHER NARCOTIC VIOLATION": 2,
        "PUBLIC PEACE VIOLATION": 2, "SEX OFFENSE": 2,
        "LIQUOR LAW VIOLATION": 2,
        "GAMBLING": 1, "PROSTITUTION": 1, "PUBLIC INDECENCY": 1,
        "OBSCENITY": 1,
        "NON-CRIMINAL": 0
    }
    return severity_map.get(type_, 1)


def engineer_features(df):
    print("✅ Starting feature engineering...")

    # --- Temporal ---
    df["Season"] = df["Month"].apply(assign_season)
    df["IsWeekend"] = df["Weekday"].isin([5, 6]).astype(int)
    df["TimeOfDay"] = pd.cut(df["Hour"], bins=[0,6,12,18,24], 
                             labels=["Night","Morning","Afternoon","Evening"], 
                             right=False)

    # --- Crime Severity ---
    df["Severity"] = df["Primary Type"].apply(create_crime_severity)

    # --- Label Encoding ---
    le_crime = LabelEncoder()
    le_loc = LabelEncoder()
    le_season = LabelEncoder()
    le_time = LabelEncoder()

    df["CrimeLabel"] = le_crime.fit_transform(df["Primary Type"])
    df["LocationLabel"] = le_loc.fit_transform(df["Location Description"].fillna("Unknown"))
    df["SeasonLabel"] = le_season.fit_transform(df["Season"])
    df["TimeLabel"] = le_time.fit_transform(df["TimeOfDay"])

    joblib.dump(le_crime, MODEL_DIR / "label_crime.pkl")
    joblib.dump(le_loc, MODEL_DIR / "label_location.pkl")
    joblib.dump(le_season, MODEL_DIR / "label_season.pkl")
    joblib.dump(le_time, MODEL_DIR / "label_time.pkl")

    # --- Geo Clustering ---
    coords = df[["Latitude","Longitude"]].dropna()
    kmeans = KMeans(n_clusters=12, random_state=42, n_init=10)
    kmeans.fit(coords)
    df["GeoCluster"] = kmeans.predict(df[["Latitude","Longitude"]].fillna(method='ffill'))
    joblib.dump(kmeans, MODEL_DIR / "geo_cluster.pkl")

    # --- Binning + Normalization ---
    df["LatBin"] = pd.qcut(df["Latitude"], 20, duplicates="drop", labels=False)
    df["LonBin"] = pd.qcut(df["Longitude"], 20, duplicates="drop", labels=False)

    scaler = MinMaxScaler()
    df[["LatNorm","LonNorm"]] = scaler.fit_transform(df[["Latitude","Longitude"]])
    joblib.dump(scaler, MODEL_DIR / "latlon_scaler.pkl")

    print("✅ Feature engineering completed")

    feature_cols = [
        "Year","Month","Day","Hour","Weekday","IsWeekend",
        "CrimeLabel","LocationLabel","SeasonLabel","TimeLabel",
        "Severity","District","Ward","Community Area",
        "GeoCluster","LatBin","LonBin","LatNorm","LonNorm","Arrest"
    ]

    out_df = df[feature_cols]
    print("✅ Final shape:", out_df.shape)
    return out_df


if __name__ == "__main__":
    print("Loading data:", DATA_PATH_01)
    df_01 = pd.read_csv(DATA_PATH_01, low_memory=False)
    print("Loading data:", DATA_PATH_01)
    df_02 = pd.read_csv(DATA_PATH_02, low_memory=False)

    df = pd.concat([df_01, df_02], ignore_index=True)
    print ("Total_rows:", len(df))
    final_df = engineer_features(df)
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Saved model-ready data: {OUTPUT_PATH}")
