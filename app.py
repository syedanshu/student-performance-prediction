from flask import Flask, render_template, request, redirect, session, jsonify
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import csv

app = Flask(__name__)
app.secret_key = "secret123"

# Load model
model = joblib.load("cgpa_model.pkl")
encoders = joblib.load("encoders.pkl")

# ---------- LOGIN ----------
@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        if request.form['username'] == "admin" and request.form['password'] == "1234":
            session['user'] = "admin"
            return redirect('/')
        return "Invalid Login"
    return render_template("login.html")

# ---------- LOGOUT ----------
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/login')

# ---------- HOME ----------
@app.route('/')
def home():
    if 'user' not in session:
        return redirect('/login')
    return render_template("index.html")

# ---------- SAVE ----------
def save_prediction(data, pred):
    with open("predictions.csv", "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data + [pred])

# ---------- PREDICT ----------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        gender = encoders["gender"].transform([request.form['gender']])[0]
        education = encoders["parental_education"].transform([request.form['education']])[0]
        college = encoders["college_name"].transform([request.form['college']])[0]
        branch = encoders["branch"].transform([request.form['branch']])[0]

        semester = float(request.form['semester'])
        study = float(request.form['study'])
        attendance = float(request.form['attendance'])
        assignments = float(request.form['assignments'])
        internal = float(request.form['internal'])

        features = np.array([[college, branch, semester, gender, education,
                              study, attendance, assignments, internal]])

        prediction = round(model.predict(features)[0], 2)

        save_prediction(list(features[0]), prediction)

        return render_template("index.html",
                               prediction_text=f"Predicted CGPA: {prediction}")

    except Exception as e:
        return f"Error: {str(e)}"

# ---------- DASHBOARD ----------
@app.route('/dashboard')
def dashboard():
    df = pd.read_excel("student_performance_prediction.xlsx")

    fig1 = px.box(df, y="average_score", color="branch",
                  title="Score Distribution")

    fig2 = px.scatter(df, x="study_hours_per_week",
                      y="average_score",
                      color="branch",
                      size="attendance_percentage",
                      title="Study vs Performance")

    fig3 = px.pie(df, names="branch", hole=0.4,
                  title="Branch Distribution")

    sem_data = df.groupby("semester")["average_score"].mean().reset_index()
    fig4 = px.line(sem_data, x="semester", y="average_score",
                   markers=True, title="Semester Trend")

    return render_template("dashboard.html",
                           graph1=fig1.to_html(False),
                           graph2=fig2.to_html(False),
                           graph3=fig3.to_html(False),
                           graph4=fig4.to_html(False))

# ---------- REAL-TIME API ----------
@app.route('/get-data')
def get_data():
    df = pd.read_excel("student_performance_prediction.xlsx")

    branch = request.args.get('branch')

    print("Selected branch:", branch)
    print("Available branches:", df['branch'].unique())

    if branch and branch != "All":
        # 🔥 FIX: make case-insensitive match
        df = df[df['branch'].str.lower() == branch.lower()]

    # If empty, fallback (IMPORTANT)
    if df.empty:
        return jsonify({
            "graph1": "<h4>No data available</h4>",
            "graph2": "",
            "graph3": "",
            "graph4": ""
        })

    fig1 = px.box(df, y="average_score", color="branch")
    fig2 = px.scatter(df, x="study_hours_per_week", y="average_score", color="branch")
    fig3 = px.pie(df, names="branch", hole=0.4)

    sem_data = df.groupby("semester")["average_score"].mean().reset_index()
    fig4 = px.line(sem_data, x="semester", y="average_score", markers=True)

    return jsonify({
        "graph1": fig1.to_html(False),
        "graph2": fig2.to_html(False),
        "graph3": fig3.to_html(False),
        "graph4": fig4.to_html(False)
    })

# ---------- RUN ----------
if __name__ == "__main__":
    app.run(debug=True)