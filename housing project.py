import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the dataset.
df = pd.read_csv('house_price_data.csv')

# Preview the data.
print(df.head())

# Convert categorical variables to numerical.
df = pd.get_dummies(df, drop_first=True)

# Define features and target variable.
X = df.drop(columns=['id', 'price'])  # Drop 'id' and the target column 'price'.
y = df['price']  # Target variable.

# Scale features.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor.
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions.
y_pred = model.predict(X_test)

# Evaluate the model.
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))
