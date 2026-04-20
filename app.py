from flask import Flask, render_template, request, redirect, session
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import csv

app = Flask(__name__)
app.secret_key = "secret123"

# Load model & encoders
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


# ---------- SAVE PREDICTIONS ----------
def save_prediction(data, pred):
    with open("predictions.csv", "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data + [pred])


# ---------- PREDICT ----------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Encode categorical inputs
        gender = encoders["gender"].transform([request.form['gender']])[0]
        education = encoders["parental_education"].transform([request.form['education']])[0]
        college = encoders["college_name"].transform([request.form['college']])[0]
        branch = encoders["branch"].transform([request.form['branch']])[0]

        # Numeric inputs
        semester = float(request.form['semester'])
        study = float(request.form['study'])
        attendance = float(request.form['attendance'])
        assignments = float(request.form['assignments'])
        internal = float(request.form['internal'])

        # ✅ EXACT SAME FEATURES AS TRAINING
        features = np.array([[college, branch, semester, gender, education,
                              study, attendance, assignments, internal]])

        prediction = model.predict(features)[0]

        # Save prediction
        save_prediction(list(features[0]), prediction)

        return render_template("index.html",
                               prediction_text=f"Predicted CGPA: {round(prediction,2)}")

    except Exception as e:
        return f"Error: {str(e)}"


# ---------- DASHBOARD ----------
@app.route('/dashboard', methods=['GET'])
def dashboard():
    df = pd.read_excel("student_performance_prediction.xlsx")

    # Get selected branch from URL
    selected_branch = request.args.get('branch')

    # Apply filter if selected
    if selected_branch and selected_branch != "All":
        df = df[df['branch'] == selected_branch]

    # Charts
    fig1 = px.histogram(df, x="average_score",
                        title="Average Score Distribution")

    fig2 = px.scatter(df,
                      x="study_hours_per_week",
                      y="average_score",
                      title="Study Hours vs Performance")

    fig3 = px.bar(df.groupby('branch')['average_score'].mean().reset_index(),
                  x='branch',
                  y='average_score',
                  title="Branch-wise Performance")

    graph1 = fig1.to_html(full_html=False)
    graph2 = fig2.to_html(full_html=False)
    graph3 = fig3.to_html(full_html=False)

    return render_template("dashboard.html",
                           graph1=graph1,
                           graph2=graph2,
                           graph3=graph3,
                           selected_branch=selected_branch)


# ---------- RUN ----------
if __name__ == "__main__":
    app.run(debug=True)