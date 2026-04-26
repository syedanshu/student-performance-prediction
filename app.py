from flask import Flask, render_template, request, redirect, session
import pandas as pd
import plotly.express as px
import joblib
import numpy as np

app = Flask(__name__)
app.secret_key = "secret123"

# Load model + encoders
model = joblib.load("cgpa_model.pkl")
encoders = joblib.load("encoders.pkl")

FILE = "student_performance_prediction.xlsx"


# ---------- LOAD DATA ----------
def load_data():
    df = pd.read_excel(FILE)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df["branch"] = df["branch"].astype(str).str.strip().str.upper()
    return df


# ---------- SAFE ENCODER ----------
def safe_encode(encoder, value):
    value = value.strip()
    classes = list(encoder.classes_)

    # Try exact match
    if value in classes:
        return encoder.transform([value])[0]

    # Try lowercase match
    for c in classes:
        if c.lower() == value.lower():
            return encoder.transform([c])[0]

    # fallback (first value)
    return encoder.transform([classes[0]])[0]


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


# ---------- PREDICT (ML MODEL) ----------
@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        return redirect('/')

    try:
        # SAFE encoding
        branch = safe_encode(encoders["branch"], request.form['branch'])
        gender = safe_encode(encoders["gender"], request.form['gender'])
        education = safe_encode(encoders["parental_education"], request.form['education'])

        semester = float(request.form['semester'])
        study = float(request.form['study'])
        attendance = float(request.form['attendance'])
        assignments = float(request.form['assignments'])
        internal = float(request.form['internal'])

        # Feature order MUST match training
        features = np.array([[
            0,  # college (dummy)
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

        return render_template("index.html",
                               prediction_text=f"🎯 Predicted CGPA: {prediction}")

    except Exception as e:
        return f"Error: {str(e)}"


# ---------- DASHBOARD ----------
@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect('/')

    df = load_data()

    branch = request.args.get("branch")
    if branch and branch != "All":
        df = df[df["branch"] == branch]

    # KPIs
    avg_score = round(df["average_score"].mean(), 2)
    max_score = round(df["average_score"].max(), 2)
    total_students = len(df)
    top_branch = df.groupby("branch")["average_score"].mean().idxmax()

    fig1 = px.box(df, y="average_score", color="branch")
    fig2 = px.scatter(df, x="study_hours_per_week", y="average_score", color="branch")
    

    sem = df.groupby("semester")["average_score"].mean().reset_index()
    fig3 = px.line(sem, x="semester", y="average_score", markers=True)

    return render_template("dashboard.html",
        g1=fig1.to_html(full_html=False),
        g2=fig2.to_html(full_html=False),
        g3=fig3.to_html(full_html=False),
        
        avg_score=avg_score,
        max_score=max_score,
        total_students=total_students,
        top_branch=top_branch,
        selected_branch=branch if branch else "All"
    )


if __name__ == "__main__":
    app.run(debug=True)