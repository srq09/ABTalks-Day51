from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib
import os


# Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target


# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)


# Create model directory
os.makedirs("model", exist_ok=True)


# Save trained model
joblib.dump(model, "model/model.pkl")

print("Model trained and saved successfully.")
print("Model path: model/model.pkl")