import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Завантаження даних
df = pd.read_csv("combined_water_quality.csv")

# Цільова змінна
target = "Kisen"
features = ["BSK5", "Fosfat", "Nitrat", "month", "Latitude", "Longitude"]

# Очищення
df = df.dropna(subset=[target] + features)
X = df[features]
y = df[target]

print(f"📦 Кількість рядків у моделі: {len(X)}")

# Розділення
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Побудова моделі
model = XGBRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = pd.Series(model.predict(X_test), index=y_test.index)

# Метрики
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\n XGBoost R² score: {r2:.4f}")
print(f" XGBoost RMSE: {rmse:.3f}")

# Обчислення залишків
residuals = y_test - y_pred
std_res = np.std(residuals)
threshold = 2 * std_res

# Виявлення аномалій
anomalies = residuals[np.abs(residuals) > threshold]
print(f"\n Виявлено аномалій: {len(anomalies)} (поріг ±{threshold:.3f})")

# Візуалізація
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.4, label="Prediction", color="blue")
plt.scatter(
    y_test.loc[anomalies.index],
    y_pred.loc[anomalies.index],
    color="red",
    label="Anomalies",
)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "k--", label="Ideal Prediction")

plt.xlabel("Observed Dissolved Oxygen")
plt.ylabel("Predicted Dissolved Oxygen")
plt.title("XGBoost Prediction Performance for Dissolved Oxygen")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
anomalies_df = X_test.loc[anomalies.index].copy()
anomalies_df["Observed_Dissolved_Oxygen"] = y_test.loc[anomalies.index]
anomalies_df["Predicted_Dissolved_Oxygen"] = y_pred.loc[anomalies.index]
anomalies_df["Residual"] = residuals.loc[anomalies.index]
anomalies_df["Post_Name"] = df.loc[anomalies.index, "Post_Name"]

anomalies_df.to_csv(
    "xgb_oxygen_anomalies.csv",
    index=False
)

print("Saved: xgb_oxygen_anomalies.csv")
