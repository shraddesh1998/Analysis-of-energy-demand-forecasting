# =========================================================
# RANDOM FOREST TIME SERIES FORECASTING
# =========================================================

# =========================================================
# IMPORT LIBRARIES
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv("final_energy_dataset.csv")

# =========================================================
# DATE CONVERSION
# =========================================================

df["Date"] = pd.to_datetime(
    df["Date"],
    dayfirst=True
)

# =========================================================
# CLEAN DEMAND
# =========================================================

df["Demand"] = pd.to_numeric(
    df["Demand"],
    errors="coerce"
)

df["Demand"] = df["Demand"].fillna(
    df["Demand"].mean()
)

# =========================================================
# SORT DATA
# =========================================================

df = df.sort_values(by="Date")

# =========================================================
# CREATE LAG FEATURES
# =========================================================

df["Lag_1"] = df["Demand"].shift(1)

df["Lag_2"] = df["Demand"].shift(2)

df["Lag_3"] = df["Demand"].shift(3)

# =========================================================
# REMOVE NULL VALUES
# =========================================================

df = df.dropna()

# =========================================================
# FEATURES
# =========================================================

X = df[
    [
        "Lag_1",
        "Lag_2",
        "Lag_3"
    ]
]

# =========================================================
# TARGET
# =========================================================

y = df["Demand"]

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

split_index = int(len(df) * 0.8)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

# =========================================================
# CREATE MODEL
# =========================================================

model = RandomForestRegressor(

    n_estimators=300,

    random_state=42

)

# =========================================================
# TRAIN MODEL
# =========================================================

model.fit(
    X_train,
    y_train
)

# =========================================================
# PREDICTIONS
# =========================================================

y_pred = model.predict(X_test)

# =========================================================
# MODEL PERFORMANCE
# =========================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)

print("\nMODEL PERFORMANCE")

print(f"MAE  : {mae:.2f}")

print(f"RMSE : {rmse:.2f}")

print(f"R2 Score : {r2:.2f}")

# =========================================================
# ACTUAL VS PREDICTED GRAPH
# =========================================================

plt.figure(figsize=(14,6))

plt.plot(
    y_test.values,
    label="Actual Demand"
)

plt.plot(
    y_pred,
    label="Predicted Demand"
)

plt.title(
    "Actual vs Predicted Demand"
)

plt.xlabel("Observations")

plt.ylabel("Demand")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()

# =========================================================
# FUTURE FORECASTING
# =========================================================

future_steps = 90

# Last 3 known demand values
last_values = list(
    df["Demand"].tail(3)
)

forecast_values = []

# =========================================================
# GENERATE FUTURE VALUES
# =========================================================

for i in range(future_steps):

    input_data = np.array(last_values[-3:]).reshape(1, -1)

    pred = model.predict(input_data)[0]

    forecast_values.append(pred)

    last_values.append(pred)

# =========================================================
# FUTURE DATES
# =========================================================

future_dates = pd.date_range(

    start=df["Date"].max() + pd.Timedelta(days=1),

    periods=future_steps

)

# =========================================================
# FORECAST DATAFRAME
# =========================================================

forecast_df = pd.DataFrame({

    "Date": future_dates,

    "Forecasted_Demand": forecast_values

})

# =========================================================
# DISPLAY FORECAST
# =========================================================

print("\nNEXT 3 MONTH FORECAST")

print(forecast_df)

# =========================================================
# SAVE FORECAST
# =========================================================

forecast_df.to_csv(

    "random_forest_forecast.csv",

    index=False

)

print("\nForecast Saved Successfully!")

# =========================================================
# FORECAST GRAPH
# =========================================================

plt.figure(figsize=(16,6))

plt.plot(
    df["Date"],
    df["Demand"],
    label="Historical Demand"
)

plt.plot(
    forecast_df["Date"],
    forecast_df["Forecasted_Demand"],
    label="Forecasted Demand"
)

plt.title(
    "Random Forest 3-Month Demand Forecast"
)

plt.xlabel("Date")

plt.ylabel("Demand")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()