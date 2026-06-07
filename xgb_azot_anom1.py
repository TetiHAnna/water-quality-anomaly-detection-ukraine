import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# Завантаження даних
df = pd.read_csv("combined_water_quality.csv")
df = df.dropna(subset=["Azot", "BSK5", "Fosfat", "Nitrat", "month", "Latitude", "Longitude", "Post_Name"])

# Ознаки та ціль
features = ["BSK5", "Fosfat", "Nitrat", "month", "Latitude", "Longitude"]
X = df[features]
y = df["Azot"]

# Розділення
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Модель
model = XGBRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = pd.Series(model.predict(X_test), index=y_test.index)

# Залишки та аномалії
residuals = y_test - y_pred
std_res = np.std(residuals)
threshold = 2 * std_res
anomalies = residuals[np.abs(residuals) > threshold]

# Побудова графіка
plt.figure(figsize=(8,6))

# Ідеальний прогноз
x_vals = np.linspace(min(y_test), max(y_test), 100)
plt.plot(x_vals, x_vals, linestyle="--", color="black", label="Ideal Prediction")

plt.scatter(y_test, y_pred, color="blue", alpha=0.5, label="Prediction")

plt.scatter(y_test.loc[anomalies.index], y_pred.loc[anomalies.index], color="red", label="Anomalies")

plt.xlabel("Observed Nitrogen Concentration")
plt.ylabel("Predicted Nitrogen Concentration")
plt.title("XGBoost Prediction Performance for Nitrogen")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
anomalies_df = X_test.loc[anomalies.index].copy()
anomalies_df["Observed_Nitrogen"] = y_test.loc[anomalies.index]
anomalies_df["Predicted_Nitrogen"] = y_pred.loc[anomalies.index]
anomalies_df["Residual"] = residuals.loc[anomalies.index]
anomalies_df["Post_Name"] = df.loc[anomalies.index, "Post_Name"]

anomalies_df.to_csv(
    "xgb_azot_anomalies.csv",
    index=False
)

print("Saved: xgb_azot_anomalies.csv")
