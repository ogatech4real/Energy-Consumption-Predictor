import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from joblib import dump

# Load your dataset
df = pd.read_csv("Scada_Data.csv")

# Features and target
X = df[['ProcTemp', 'EnvTemp']]
y = df['EnergyCon']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
dump(model, 'energy_predictor.joblib')

print("Model trained and saved.")