import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Завантаження даних
df = pd.read_csv("combined_water_quality.csv")
df = df.dropna(subset=["Kisen", "BSK5", "Fosfat", "Nitrat", "month", "Latitude", "Longitude", "Post_Name"])

# Ознаки та ціль
features = ["BSK5", "Fosfat", "Nitrat", "month", "Latitude", "Longitude"]
X = df[features]
y = df["Kisen"]

# Розділення
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Модель
model = RandomForestRegressor(n_estimators=100, random_state=42)
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

# Прогнозовані точки
plt.scatter(y_test, y_pred, color="blue", alpha=0.5, label="Prediction")

# Аномалії
plt.scatter(y_test.loc[anomalies.index], y_pred.loc[anomalies.index], color="red", label="Anomalies")

plt.xlabel("Observed Dissolved Oxygen")
plt.ylabel("Predicted Dissolved Oxygen")
plt.title("Random Forest Prediction Performance for Dissolved Oxygen")
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
    "rf_oxygen_anomalies.csv",
    index=False
)

print("Saved: rf_oxygen_anomalies.csv")
