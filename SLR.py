import pandas as pd
import numpy as np
df = pd.read_csv("C:/Users/shraddhesh.datkhile/Downloads/open-meteo-51.49N10.43E309m (2).csv")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())
df.columns = [
    "Date",
    "Temperature_Mean",
    "Temperature_Max",
    "Precipitation",
    "Sunshine",
    "Wind_Speed",
    "Consumption",
    "Demand"
]
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
numeric_columns = df.select_dtypes(include=np.number).columns
for col in numeric_columns:
    df[col] = df[col].fillna(df[col].mean())
df = df.sort_values(by="Date")
df = df.reset_index(drop=True)
print("\nCleaned Dataset Info:")
print(df.info())
print("\nFinal Missing Values:")
print(df.isnull().sum())
df.to_csv("cleaned_energy_data.csv", index=False)