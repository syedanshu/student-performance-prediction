import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
df = pd.read_excel("student_performance_prediction.xlsx")

# Encode categorical columns
encoders = {}
for col in ["gender", "parental_education", "college_name", "branch"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Create CGPA
df["cgpa"] = df["average_score"] / 10

# Features & target
X = df.drop(["USN", "average_score", "result", "cgpa"], axis=1)
y = df["cgpa"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "cgpa_model.pkl")
joblib.dump(encoders, "encoders.pkl")

print("✅ Model trained successfully")