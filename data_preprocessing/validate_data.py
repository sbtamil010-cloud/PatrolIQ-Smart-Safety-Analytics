# src/data_preprocessing/data_validation.py
"""
data_validation.py
------------------------------------
Performs data quality and validation checks
for the Chicago Crime dataset used in PatrolIQ.
Generates both text and JSON reports for UI display.
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from io import StringIO

# --- Define Base Paths ---
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH_01 = BASE_DIR / "data" / "processed" / "sample_250000_rows_01.csv"
DATA_PATH_02 = BASE_DIR / "data" / "processed" / "sample_250000_rows_02.csv"
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def validate_data(DATA_PATH_01, DATA_PATH_02) -> None:
    """Run full data quality assessment and save reports."""
    buffer = StringIO()
    def log(msg):  # helper to print + store text
        print(msg)
        buffer.write(msg + "\n")

    log("🔍 Starting Data Quality Validation")
    log("-" * 60)

    summary = {}  # JSON summary output

    # 1️⃣ Load data
    try:
        print("Loading data:", DATA_PATH_01)
        df_01 = pd.read_csv(DATA_PATH_01, low_memory=False)
        print("Loading data:", DATA_PATH_01)
        df_02 = pd.read_csv(DATA_PATH_02, low_memory=False)

        df = pd.concat([df_01, df_02], ignore_index=True)
        print("Total_rows:", len(df))

        log(f"✅ Loaded dataset: {len(df):,} records, {len(df.columns)} columns")
        summary["records"] = len(df)
        summary["columns"] = len(df.columns)
    except Exception as e:
        log(f"❌ Error loading data: {e}")
        return

    # 2️⃣ Missing values
    missing = df.isna().sum()
    missing_cols = missing[missing > 0]
    summary["missing_values"] = missing_cols.to_dict()
    if not missing_cols.empty:
        log("\n🧩 Missing values found:")
        log(str(missing_cols))
    else:
        log("\n✅ No missing values detected")

    # 3️⃣ Duplicates
    dup_count = df.duplicated().sum()
    log(f"\n📦 Duplicate rows: {dup_count}")
    summary["duplicates"] = int(dup_count)

    # 4️⃣ Data types
    log("\n🧱 Data types summary:")
    log(str(df.dtypes.astype(str).to_dict()))

    # 5️⃣ Date validation
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        invalid_dates = df[df["Date"] > pd.Timestamp.today()]
        log(f"\n⏰ Future dates found: {len(invalid_dates)}")
        log(f"📅 Date range: {df['Date'].min()} → {df['Date'].max()}")
        summary["date_range"] = [str(df["Date"].min()), str(df["Date"].max())]

    # 6️⃣ Year range
    if "Year" in df.columns:
        log(f"Year range: {df['Year'].min()} → {df['Year'].max()}")
        summary["year_range"] = [int(df["Year"].min()), int(df["Year"].max())]

    # 7️⃣ Geographic validation
    if {"Latitude", "Longitude"}.issubset(df.columns):
        geo_outliers = df[
            (df["Latitude"] < 41.4) | (df["Latitude"] > 42.1) |
            (df["Longitude"] < -87.95) | (df["Longitude"] > -87.5)
        ]
        log(f"\n🗺️ Out-of-bound coordinates: {len(geo_outliers)}")
        summary["geo_outliers"] = int(len(geo_outliers))

    # 8️⃣ Crime category validation
    if "Primary Type" in df.columns:
        valid_crimes = [
            'OFFENSE INVOLVING CHILDREN', 'THEFT', 'MOTOR VEHICLE THEFT',
            'CRIMINAL TRESPASS', 'OTHER OFFENSE', 'CRIMINAL DAMAGE', 'ASSAULT', 'BATTERY',
            'BURGLARY', 'NARCOTICS', 'DECEPTIVE PRACTICE', 'ROBBERY', 'KIDNAPPING',
            'PUBLIC PEACE VIOLATION', 'WEAPONS VIOLATION', 'ARSON', 'PROSTITUTION',
            'CRIM SEXUAL ASSAULT', 'LIQUOR LAW VIOLATION',
            'INTERFERENCE WITH PUBLIC OFFICER', 'SEX OFFENSE',
            'CRIMINAL SEXUAL ASSAULT', 'HOMICIDE', 'GAMBLING', 'INTIMIDATION', 'STALKING',
            'CONCEALED CARRY LICENSE VIOLATION', 'OBSCENITY', 'HUMAN TRAFFICKING',
            'PUBLIC INDECENCY', 'OTHER NARCOTIC VIOLATION', 'NON-CRIMINAL'
        ]
        invalid_types = df[~df["Primary Type"].isin(valid_crimes)]
        log(f"🚫 Invalid crime categories: {len(invalid_types)}")
        summary["invalid_crime_types"] = int(len(invalid_types))

    # 9️⃣ Temporal coverage
    for col in ["Hour", "Month", "Day"]:
        if col in df.columns:
            log(f"{col} range: {df[col].min()} → {df[col].max()}")
            summary[f"{col.lower()}_range"] = [int(df[col].min()), int(df[col].max())]

    # 🔟 Integrity checks
    if {"Arrest", "Case Number"}.issubset(df.columns):
        invalid_arrests = df[(df["Arrest"] == True) & (df["Case Number"].isna())]
        log(f"\n⚠️ Arrests without Case Number: {len(invalid_arrests)}")
        summary["invalid_arrests"] = int(len(invalid_arrests))

    log("\n✅ Data quality validation completed successfully!")
    log("-" * 60)

    # --- Save Reports ---
    txt_path = REPORT_DIR / "data_validation_report.txt"
    json_path = REPORT_DIR / "data_validation_summary.json"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(buffer.getvalue())

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print(f"📝 Reports saved to:\n  {txt_path}\n  {json_path}")


if __name__ == "__main__":
    validate_data(DATA_PATH_01, DATA_PATH_02)
