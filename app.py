from flask import Flask, render_template, request, redirect, session
import pandas as pd
import plotly.express as px
import joblib
import numpy as np

app = Flask(__name__)
app.secret_key = "secret123"

# Load model
model = joblib.load("cgpa_model.pkl")
encoders = joblib.load("encoders.pkl")

FILE = "student_performance_prediction.xlsx"

feature_names = [
    "college",
    "branch",
    "semester",
    "gender",
    "education",
    "study_hours",
    "attendance",
    "assignments",
    "internal_marks"
]

# ---------- SAFE ENCODER ----------
def safe_encode(encoder, value):
    value = value.strip()
    classes = list(encoder.classes_)

    if value in classes:
        return encoder.transform([value])[0]

    for c in classes:
        if c.lower() == value.lower():
            return encoder.transform([c])[0]

    return encoder.transform([classes[0]])[0]


# ---------- FEEDBACK ----------
def generate_feedback(study, attendance, assignments, internal, cgpa):
    feedback = []

    if cgpa >= 8:
        feedback.append("🌟 Excellent performance! Keep it up.")
    elif cgpa >= 6:
        feedback.append("👍 Good performance, improve consistency.")
    else:
        feedback.append("⚠️ Performance needs improvement.")

    if study < 4:
        feedback.append("📚 Increase study hours.")
    if attendance < 75:
        feedback.append("📝 Improve attendance.")
    if assignments < 5:
        feedback.append("📂 Complete more assignments.")
    if internal < 25:
        feedback.append("📊 Improve internal marks.")

    return feedback


# ---------- LOGIN ----------
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form['username'] == "admin" and request.form['password'] == "1234":
            session['user'] = "admin"
            return redirect('/cgpa')
        return "Invalid Login"
    return render_template("login.html")


# ---------- LOGOUT ----------
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/')


# ---------- CGPA PAGE ----------
@app.route('/cgpa')
def cgpa():
    if 'user' not in session:
        return redirect('/')
    return render_template("index.html")


# ---------- PREDICT ----------
@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        return redirect('/')

    try:
        # Encode
        branch = safe_encode(encoders["branch"], request.form['branch'])
        gender = safe_encode(encoders["gender"], request.form['gender'])
        education = safe_encode(encoders["parental_education"], request.form['education'])

        semester = float(request.form['semester'])
        study = float(request.form['study'])
        attendance = float(request.form['attendance'])
        assignments = float(request.form['assignments'])
        internal = float(request.form['internal'])

        features = np.array([[
            0,
            branch,
            semester,
            gender,
            education,
            study,
            attendance,
            assignments,
            internal
        ]])

        prediction = round(model.predict(features)[0], 2)

        # Feedback
        feedback = generate_feedback(study, attendance, assignments, internal, prediction)

        # ---------- FEATURE IMPORTANCE ----------
        importances = model.feature_importances_
        importance_map = dict(zip(feature_names, importances))

        def impact_label(val):
            if val > 0.3:
                return "🔥 High impact"
            elif val > 0.1:
                return "⚡ Moderate impact"
            else:
                return "🔹 Low impact"

        # ---------- HUMAN READABLE INSIGHTS ----------
        insights = [
            f"📚 Study Hours: {study} hrs/week ({impact_label(importance_map['study_hours'])})",
            f"📝 Attendance: {attendance}% ({impact_label(importance_map['attendance'])})",
            f"📂 Assignments: {assignments} ({impact_label(importance_map['assignments'])})",
            f"📊 Internal Marks: {internal}/100 ({impact_label(importance_map['internal_marks'])})",
            f"📅 Semester: {int(semester)} ({impact_label(importance_map['semester'])})"
        ]

        # ---------- AI EXPLANATION ----------
        explanation = []

        if importance_map["study_hours"] > 0.2:
            explanation.append("📚 Study hours significantly influence CGPA.")

        if importance_map["internal_marks"] > 0.2:
            explanation.append("📊 Internal marks are a major factor in performance.")

        if importance_map["attendance"] > 0.1:
            explanation.append("📝 Attendance contributes to consistency and results.")

        if importance_map["assignments"] < 0.05:
            explanation.append("📂 Assignments have relatively low impact.")

        return render_template(
            "index.html",
            prediction_text=f"🎯 Predicted CGPA: {prediction}",
            feedback=feedback,
            insights=insights,
            explanation=explanation
        )

    except Exception as e:
        return f"Error: {str(e)}"


# ---------- DASHBOARD ----------
@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect('/')

    df = pd.read_excel(FILE)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    fig1 = px.box(df, y="average_score", color="branch")

    return render_template("dashboard.html",
                           g1=fig1.to_html(full_html=False))


if __name__ == "__main__":
    app.run(debug=True)