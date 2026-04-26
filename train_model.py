import pandas as pd
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

# ✅ FIX 1: Proper CGPA calculation
df["cgpa"] = df["average_score"] / 10

# ✅ FIX 2: Limit CGPA range (VERY IMPORTANT)
df["cgpa"] = df["cgpa"].clip(4, 10)

# Features
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

# ✅ FIX 3: Better model (prevents overfitting)
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42
)

model.fit(X, y)

# Save model
joblib.dump(model, "cgpa_model.pkl")
joblib.dump(encoders, "encoders.pkl")

print("✅ Model retrained with realistic CGPA")