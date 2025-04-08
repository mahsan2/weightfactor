import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import lime.lime_tabular

# Load dataset
df = pd.read_excel("Copy of Test file -ZS.xlsx", header=2)
df.columns = ['Sample', 'Factor_Weight', 'A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12']

# Clean currency and percentage symbols
for col in ['A5', 'A6']:
    df[col] = df[col].replace('[\$,]', '', regex=True).astype(float)
for col in ['A3', 'A4', 'A10', 'A11']:
    df[col] = df[col].replace('%','',regex=True).astype(float)

# A10 inversion
df['A10'] = df['A10'].max() - df['A10']
a10_max_value = df['A10'].max()

# Split
X = df[['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12']]
y = df['Factor_Weight']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.2f}")

# Save everything
joblib.dump({
    'model': model,
    'scaler': scaler,
    'a10_max_value': a10_max_value,
    'feature_names': X.columns.tolist(),
    'X_train_scaled': X_train_scaled
}, 'factor_weight_model.pkl')

print("✅ Model and supporting objects saved to factor_weight_model.pkl")
