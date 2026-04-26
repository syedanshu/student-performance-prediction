from flask import Flask, render_template, request, redirect, session, jsonify
import pandas as pd
import plotly.express as px

app = Flask(__name__)
app.secret_key = "secret123"

DATA_PATH = "student_performance_prediction.xlsx"

# ---------- CLEAN DATA ----------
def clean_data(df):
    # normalize columns
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # flexible mapping
    col_map = {
        "average score": "average_score",
        "study hours": "study_hours_per_week",
        "attendance": "attendance_percentage"
    }
    df = df.rename(columns=col_map)

    # ensure required columns exist
    # fallback: pick first numeric column if average_score missing
    if "average_score" not in df.columns:
        num_cols = df.select_dtypes(include="number").columns
        if len(num_cols) > 0:
            df["average_score"] = df[num_cols[0]]

    if "branch" not in df.columns:
        df["branch"] = "UNKNOWN"

    df["branch"] = df["branch"].astype(str).str.strip().str.upper()

    return df


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


# ---------- DASHBOARD ----------
@app.route('/dashboard')
def dashboard():
    df = pd.read_excel(DATA_PATH)
    df = clean_data(df)

    fig1 = px.box(df, y="average_score", color="branch")
    fig2 = px.scatter(df, x=df.columns[0], y="average_score", color="branch")
    fig3 = px.pie(df, names="branch")

    # safe semester fallback
    if "semester" in df.columns:
        sem = df.groupby("semester")["average_score"].mean().reset_index()
        fig4 = px.line(sem, x="semester", y="average_score", markers=True)
    else:
        fig4 = px.line(df.head(10), y="average_score")

    return render_template("dashboard.html",
        g1=fig1.to_json(),
        g2=fig2.to_json(),
        g3=fig3.to_json(),
        g4=fig4.to_json()
    )


# ---------- FILTER API ----------
@app.route('/get-data')
def get_data():
    df = pd.read_excel(DATA_PATH)
    df = clean_data(df)

    branch = request.args.get("branch")

    if branch and branch != "All":
        branch = branch.strip().upper()
        df = df[df["branch"] == branch]

    if df.empty:
        return jsonify({"empty": True})

    fig1 = px.box(df, y="average_score", color="branch")
    fig2 = px.scatter(df, x=df.columns[0], y="average_score", color="branch")
    fig3 = px.pie(df, names="branch")

    if "semester" in df.columns:
        sem = df.groupby("semester")["average_score"].mean().reset_index()
        fig4 = px.line(sem, x="semester", y="average_score", markers=True)
    else:
        fig4 = px.line(df.head(10), y="average_score")

    return jsonify({
        "empty": False,
        "g1": fig1.to_json(),
        "g2": fig2.to_json(),
        "g3": fig3.to_json(),
        "g4": fig4.to_json()
    })


if __name__ == "__main__":
    app.run(debug=True)