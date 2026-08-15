# src/data_preprocessing/clean_data.py
import pandas as pd
import numpy as np
from pathlib import Path

# ======================================================
# PatrolIQ - Data Cleaning & Sampling Script
# Works both locally and in cloud deployment
# ======================================================

# --- Define Base Paths ---
BASE_DIR = Path(__file__).resolve().parents[2]  # goes two levels up to project root
RAW_PATH = BASE_DIR / "data" / "raw" / "Crimes_-_2001_to_Present_20251025.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "sample_500000_rows.csv"

# --- Load Dataset ---
print(f"📂 Loading dataset from: {RAW_PATH}")
df = pd.read_csv(RAW_PATH)

# --- Clean Missing and Duplicate Records ---
df = df.dropna().drop_duplicates(keep="last")



# --- Handle Numeric Outliers (IQR Clipping) ---
num_cols = df.select_dtypes(include=[np.number]).columns
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = np.clip(df[col], lower, upper)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"])
df = df[df["Date"].dt.year >= 2010]
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["Weekday"] = df["Date"].dt.weekday
df["Hour"] = df["Date"].dt.hour
df["Minute"] = df["Date"].dt.minute
time_groups = df.groupby(["Year", "Month", "Day", "Hour"], group_keys=False)
core_sample = time_groups.apply(lambda x: x.sample(n=1, random_state=42))
print("Core sample size:", len(core_sample))
remaining_needed = 500000 - len(core_sample)
if remaining_needed > 0:
    remaining_df = df.drop(core_sample.index)
    extra_sample = remaining_df.sample(n=remaining_needed, random_state=42)
    final_sample = pd.concat([core_sample, extra_sample])
else:
    final_sample = core_sample

print("Final sample size:", len(final_sample))



# --- Save Processed Dataset ---
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
final_sample.to_csv(OUTPUT_PATH, index=False)


print(f"✅ Cleaned and sampled dataset saved to: {OUTPUT_PATH}")
