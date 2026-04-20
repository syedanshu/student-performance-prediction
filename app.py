from flask import Flask, render_template, request, redirect, session
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv

app = Flask(__name__)
app.secret_key = "secret123"

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
    gender = encoders["gender"].transform([request.form['gender']])[0]
    edu = encoders["parental_education"].transform([request.form['education']])[0]
    college = encoders["college_name"].transform([request.form['college']])[0]
    branch = encoders["branch"].transform([request.form['branch']])[0]

    semester = float(request.form['semester'])
    study = float(request.form['study'])
    attendance = float(request.form['attendance'])
    assignments = float(request.form['assignments'])
    internal = float(request.form['internal'])

    features = np.array([[college, branch, semester, gender, edu,
                          study, attendance, assignments, internal]])

    pred = model.predict(features)[0]

    save_prediction(list(features[0]), pred)

    return render_template("index.html",
                           prediction_text=f"Predicted CGPA: {round(pred,2)}")

# ---------- DASHBOARD ----------
@app.route('/dashboard')
def dashboard():
    df = pd.read_excel("student_performance_prediction.xlsx")

    if not os.path.exists("static"):
        os.makedirs("static")

    plt.figure()
    df['average_score'].hist()
    plt.savefig("static/chart1.png")
    plt.close()

    plt.figure()
    plt.scatter(df['study_hours_per_week'], df['average_score'])
    plt.savefig("static/chart2.png")
    plt.close()

    plt.figure()
    df.groupby('branch')['average_score'].mean().plot(kind='bar')
    plt.savefig("static/chart3.png")
    plt.close()

    return render_template("dashboard.html")

if __name__ == "__main__":
    app.run(debug=True)