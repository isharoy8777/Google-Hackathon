import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Step 1: Load and Preprocess Data
# Assuming you have a dataset in a CSV file with features and target (combinational depth)
# Columns: feature1, feature2, ..., featureN, combinational_depth
data = pd.read_csv("rtl_dataset.csv")

# Features (X) and Target (y)
X = data.drop("combinational_depth", axis=1)  # Features
y = data["combinational_depth"]  # Target (combinational depth)

# Step 2: Feature Scaling (Normalization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 4: Train the Model (Random Forest Regressor)
model = RandomForestRegressor(n_estimators=100, random_state=42)  # 100 trees
model.fit(X_train, y_train)

# Step 5: Evaluate the Model
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2): {r2}")

# Step 6: Predict Combinational Depth for New Signals
# Example: New signal features (scaled)
new_signal_features = np.array([[fan_in, fan_out, path_length, ...]])  # Replace with actual values
new_signal_features_scaled = scaler.transform(new_signal_features)

predicted_depth = model.predict(new_signal_features_scaled)
print(f"Predicted Combinational Depth: {predicted_depth[0]}")
