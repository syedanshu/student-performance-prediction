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

# ✅ ONLY SELECT REQUIRED FEATURES (9 features)
X = df[[
    "college_name",
    "branch",
    "semester",
    "gender",
    "parental_education",
    "study_hours_per_week",
    "attendance_percentage",
    "assignments_completed",
    "internal_marks"
]]

y = df["cgpa"]

# Train model
model = RandomForestRegressor()
model.fit(X, y)

# Save
joblib.dump(model, "cgpa_model.pkl")
joblib.dump(encoders, "encoders.pkl")

print("✅ Model trained successfully")