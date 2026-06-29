"""
ML Health — disease prediction web app.

Flask app serving three scikit-learn models (heart, obesity, symptoms) behind a
multi-page UI, plus user accounts, a saved health-check history, a dashboard,
a health-analysis page, and a rule-based health assistant chatbot.

Models load from models/*.pkl. Loading is graceful: a missing model never
crashes the app — that predictor just reports "unavailable" and /health shows
what loaded.

Not medical advice — an educational demo.

Run locally:
    python app.py        # http://127.0.0.1:5003
"""
import json
import os
import sqlite3
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

import numpy as np
from flask import (
    Flask, flash, g, jsonify, redirect, render_template, request, session,
    url_for,
)
from flask_cors import CORS
from flask_wtf import CSRFProtect
from werkzeug.security import check_password_hash, generate_password_hash

try:
    import joblib
except ImportError:
    joblib = None

try:
    from train_symptoms import DISEASE_SYMPTOMS
except Exception:  # noqa: BLE001
    DISEASE_SYMPTOMS = {}

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DB_PATH = BASE_DIR / "mlhealth.db"

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-only-mlhealth-key")
CORS(app)
csrf = CSRFProtect(app)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def load_model(filename):
    path = MODELS_DIR / filename
    if joblib is None or not path.exists():
        app.logger.warning("Model not loaded: %s", path)
        return None
    try:
        return joblib.load(path)
    except Exception as exc:  # noqa: BLE001
        app.logger.warning("Failed to load %s: %s", path, exc)
        return None


heart_model = load_model("heart_model.pkl")
obesity_model = load_model("obesity_model.pkl")
symptoms_model = load_model("symptoms_model.pkl")

SYMPTOM_VOCAB = []
_vocab_path = MODELS_DIR / "symptoms_columns.json"
if _vocab_path.exists():
    try:
        SYMPTOM_VOCAB = json.loads(_vocab_path.read_text())
    except Exception as exc:  # noqa: BLE001
        app.logger.warning("Failed to read symptom vocabulary: %s", exc)


def normalise_symptom(text):
    return text.strip().lower().replace(" ", "_")


def model_unavailable(name):
    return jsonify(
        error=True,
        prediction=f"The {name} predictor isn't available in this demo yet.",
    ), 503


# --------------------------------------------------------------------------- #
# Database (users + saved health checks)
# --------------------------------------------------------------------------- #
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                name       TEXT NOT NULL,
                username   TEXT NOT NULL UNIQUE,
                pwd_hash   TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS checks (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    INTEGER NOT NULL,
                kind       TEXT NOT NULL,
                summary    TEXT NOT NULL,
                result     TEXT NOT NULL,
                flagged    INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            """
        )


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def current_user():
    uid = session.get("user_id")
    if uid is None:
        return None
    return get_db().execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if current_user() is None:
            flash("Please log in to view that.", "error")
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


def record_check(kind, summary, result, flagged=False):
    """Save a prediction to the logged-in user's history (no-op if logged out)."""
    user = current_user()
    if user is None:
        return
    get_db().execute(
        "INSERT INTO checks (user_id, kind, summary, result, flagged, created_at) "
        "VALUES (?,?,?,?,?,?)",
        (user["id"], kind, summary, result, int(flagged), now_iso()),
    )
    get_db().commit()


@app.context_processor
def inject_user():
    return {"user": current_user()}


@app.template_filter("nice_dt")
def nice_dt(value):
    try:
        return datetime.fromisoformat(value).strftime("%d %b %Y, %H:%M")
    except (ValueError, TypeError):
        return value


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/heart")
def heart_page():
    return render_template("heart.html")


@app.route("/obesity")
def obesity_page():
    return render_template("obesity.html")


@app.route("/symptoms")
def symptoms_page():
    return render_template("symptoms.html", symptom_vocab=SYMPTOM_VOCAB)


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "GET":
        return render_template("signup.html")
    db = get_db()
    f = request.form
    name = f.get("name", "").strip()
    username = f.get("username", "").strip()
    pwd = f.get("password", "")
    confirm = f.get("confirm", "")

    if not name:
        flash("Please enter your name.", "error")
    elif len(username) < 3:
        flash("Username must be at least 3 characters.", "error")
    elif len(pwd) < 6:
        flash("Password must be at least 6 characters.", "error")
    elif pwd != confirm:
        flash("Passwords do not match.", "error")
    elif db.execute("SELECT 1 FROM users WHERE username=?", (username,)).fetchone():
        flash("That username is taken.", "error")
    else:
        db.execute(
            "INSERT INTO users (name, username, pwd_hash, created_at) VALUES (?,?,?,?)",
            (name, username, generate_password_hash(pwd), now_iso()),
        )
        db.commit()
        flash("Account created — please log in.", "success")
        return redirect(url_for("login"))
    return render_template("signup.html", form=f), 400


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html")
    db = get_db()
    username = request.form.get("username", "").strip()
    pwd = request.form.get("password", "")
    row = db.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    if row and check_password_hash(row["pwd_hash"], pwd):
        session.clear()
        session["user_id"] = row["id"]
        flash(f"Welcome, {row['name']}.", "success")
        return redirect(url_for("dashboard"))
    flash("Invalid username or password.", "error")
    return render_template("login.html"), 400


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out.", "success")
    return redirect(url_for("index"))


# --------------------------------------------------------------------------- #
# Predictions (CSRF-exempt JSON APIs; save history when logged in)
# --------------------------------------------------------------------------- #
def _bin_features(age, impulse, sys_bp, dia_bp, glucose, kcm, troponin):
    return [
        0 if age < 40 else 1,
        0 if impulse < 70 else 1,
        0 if sys_bp < 120 else 1,
        0 if dia_bp < 80 else 1,
        0 if glucose < 100 else 1,
        0 if kcm < 5 else 1,
        0 if troponin < 0.04 else 1,
    ]


@app.route("/predict/heart", methods=["POST"])
@csrf.exempt
def predict_heart():
    if heart_model is None:
        return model_unavailable("heart")
    try:
        d = request.get_json(force=True)
        raw = [float(d["age"]), float(d["gender"]), float(d["impluse"]),
               float(d["pressurehight"]), float(d["pressurelow"]),
               float(d["glucose"]), float(d["kcm"]), float(d["troponin"])]
        features = np.array([raw + _bin_features(raw[0], raw[2], raw[3], raw[4],
                                                 raw[5], raw[6], raw[7])])
        pred = int(heart_model.predict(features)[0])
        result = "Positive" if pred == 1 else "Negative"
        record_check("heart",
                     f"Age {int(raw[0])}, BP {int(raw[3])}/{int(raw[4])}, "
                     f"glucose {int(raw[5])}, troponin {raw[7]}",
                     result, flagged=(pred == 1))
        return jsonify(prediction=result)
    except Exception as exc:  # noqa: BLE001
        return jsonify(error=True, prediction=f"Could not predict: {exc}"), 400


@app.route("/predict/obesity", methods=["POST"])
@csrf.exempt
def predict_obesity():
    if obesity_model is None:
        return model_unavailable("obesity")
    try:
        d = request.get_json(force=True)
        age = float(d["age"]); gender = float(d["gender"])
        height = float(d["height"]); weight = float(d["weight"])
        bmi = weight / ((height / 100) ** 2)
        label = str(obesity_model.predict(np.array([[age, gender, height, weight, bmi]]))[0])
        record_check("obesity", f"BMI {bmi:.1f} ({int(height)}cm, {int(weight)}kg)",
                     label, flagged=("obese" in label.lower()))
        return jsonify(prediction=label)
    except ZeroDivisionError:
        return jsonify(error=True, prediction="Height must be greater than zero."), 400
    except Exception as exc:  # noqa: BLE001
        return jsonify(error=True, prediction=f"Could not predict: {exc}"), 400


@app.route("/predict/symptoms", methods=["POST"])
@csrf.exempt
def predict_symptoms():
    if symptoms_model is None or not SYMPTOM_VOCAB:
        return model_unavailable("symptoms")
    try:
        d = request.get_json(force=True)
        text = str(d.get("symptom", ""))
        tokens = {normalise_symptom(t) for t in text.split(",") if t.strip()}
        vector = [1 if s in tokens else 0 for s in SYMPTOM_VOCAB]
        if sum(vector) == 0:
            return jsonify(
                error=True,
                prediction="None of those match the recognised symptoms — "
                           "pick from the listed ones.",
            ), 400
        disease = str(symptoms_model.predict(np.array([vector]))[0])
        record_check("symptoms", text.strip()[:120], disease)
        return jsonify(prediction=disease)
    except Exception as exc:  # noqa: BLE001
        return jsonify(error=True, prediction=f"Could not predict: {exc}"), 400


# --------------------------------------------------------------------------- #
# Dashboard + health analysis
# --------------------------------------------------------------------------- #
@app.route("/dashboard")
@login_required
def dashboard():
    db = get_db()
    user = current_user()
    checks = db.execute(
        "SELECT * FROM checks WHERE user_id=? ORDER BY id DESC LIMIT 10", (user["id"],)
    ).fetchall()
    counts = {"heart": 0, "obesity": 0, "symptoms": 0}
    for r in db.execute(
        "SELECT kind, COUNT(*) c FROM checks WHERE user_id=? GROUP BY kind",
        (user["id"],),
    ).fetchall():
        counts[r["kind"]] = r["c"]
    flagged = db.execute(
        "SELECT COUNT(*) c FROM checks WHERE user_id=? AND flagged=1", (user["id"],)
    ).fetchone()["c"]
    total = sum(counts.values())
    return render_template("dashboard.html", checks=checks, counts=counts,
                           total=total, flagged=flagged)


def _bars(totals):
    top = max(totals.values(), default=0) or 1
    return [{"label": k, "value": v, "pct": round(v / top * 100)}
            for k, v in sorted(totals.items(), key=lambda kv: -kv[1])]


@app.route("/analysis")
@login_required
def analysis():
    db = get_db()
    user = current_user()
    rows = db.execute("SELECT * FROM checks WHERE user_id=?", (user["id"],)).fetchall()

    by_kind, results = {}, {}
    for r in rows:
        by_kind[r["kind"]] = by_kind.get(r["kind"], 0) + 1
        results.setdefault(r["kind"], {})
        results[r["kind"]][r["result"]] = results[r["kind"]].get(r["result"], 0) + 1

    heart_results = _bars(results.get("heart", {}))
    obesity_results = _bars(results.get("obesity", {}))
    symptom_results = _bars(results.get("symptoms", {}))

    return render_template(
        "analysis.html",
        count=len(rows),
        kinds=_bars(by_kind),
        heart_results=heart_results,
        obesity_results=obesity_results,
        symptom_results=symptom_results,
    )


# --------------------------------------------------------------------------- #
# Health assistant chatbot (rule-based; educational, not medical advice)
# --------------------------------------------------------------------------- #
NORMAL_RANGES = {
    "blood pressure": "Normal blood pressure is around 120/80 mmHg. 130/80 or "
                      "higher is considered high (hypertension).",
    "heart rate": "A normal resting heart rate is about 60–100 beats per minute.",
    "glucose": "Fasting blood sugar: normal is under 100 mg/dL, 100–125 is "
               "pre-diabetes, and 126+ suggests diabetes.",
    "temperature": "Normal body temperature is about 36.5–37.5°C (97.7–99.5°F); "
                   "above ~38°C is a fever.",
    "troponin": "Troponin is usually under ~0.04 ng/mL; higher levels can "
                "indicate heart-muscle damage.",
    "bmi": "BMI categories: under 18.5 underweight, 18.5–24.9 normal, "
           "25–29.9 overweight, 30+ obese. BMI = weight(kg) / height(m)².",
}
DISCLAIMER = ("\n\n(I'm a demo assistant, not a doctor — for real concerns, "
              "please see a healthcare professional.)")


def bot_reply(message):
    msg = " ".join(message.lower().split())
    tokens = set(msg.replace(",", " ").split())

    def has(*w):
        return any(x in msg for x in w)

    def word(*w):
        return any(x in tokens for x in w)

    if not msg:
        return "Ask me about symptoms, healthy ranges (BP, glucose, BMI…), or which check to use."

    # safety first
    if has("chest pain", "can't breathe", "cant breathe", "difficulty breathing",
           "severe bleeding", "unconscious", "suicid", "heart attack", "stroke"):
        return ("That could be a medical emergency. Please contact your local "
                "emergency services or get to a hospital right away.")

    if word("hi", "hello", "hey") or has("good morning", "good afternoon", "good evening"):
        return ("Hi! I'm the ML Health assistant. I can explain healthy ranges "
                "(blood pressure, glucose, BMI…), suggest what symptoms might "
                "relate to, and point you to the right check." + DISCLAIMER)

    # normal ranges
    for key, answer in NORMAL_RANGES.items():
        if key in msg or (key == "blood pressure" and has("bp", "pressure")) \
                or (key == "glucose" and has("sugar", "diabet")) \
                or (key == "heart rate" and has("pulse", "bpm")):
            return answer + DISCLAIMER

    # "what could cause <symptoms>" — match against the knowledge base
    if has("cause", "could it be", "what is wrong", "symptom of", "mean", "have"):
        matches = _match_conditions(msg)
        if matches:
            names = ", ".join(matches)
            return (f"Symptoms like that can be associated with: {names}. "
                    "Use the Symptoms checker for a model-based guess." + DISCLAIMER)

    # which check
    if has("which", "what test", "what check", "how do i", "where do i"):
        return ("Use Heart for cardiac risk (vitals + blood markers), Obesity for "
                "BMI-based classification, and Symptoms to guess a condition from "
                "symptoms. Pick one from the top menu.")
    if has("heart", "cardiac"):
        return ("The Heart check estimates heart-disease risk from age, vitals, "
                "blood sugar and markers like troponin. Open the Heart page to try it." + DISCLAIMER)
    if has("obesity", "weight", "bmi"):
        return ("The Obesity check classifies weight status from your age, "
                "height and weight (it computes BMI). Open the Obesity page." + DISCLAIMER)
    if has("symptom"):
        return ("On the Symptoms page, add your symptoms (click the chips) and "
                "I'll guess a likely condition with the model." + DISCLAIMER)

    # generic symptom mention
    matches = _match_conditions(msg)
    if matches:
        return (f"Those symptoms may relate to: {', '.join(matches)}. Try the "
                "Symptoms checker to see the model's prediction." + DISCLAIMER)

    if has("help", "what can you", "menu", "option"):
        return ("I can: explain healthy ranges (BP, heart rate, glucose, BMI, "
                "troponin, temperature), tell you which check to use, and relate "
                "symptoms to possible conditions. What would you like?")
    if word("thanks", "thank"):
        return "You're welcome — stay well!"
    if word("bye", "goodbye"):
        return "Take care!"
    return ("I can help with healthy ranges, choosing a check, or relating "
            "symptoms to conditions. Try: \"what's a normal blood pressure?\" or "
            "\"what could cause fever and cough?\"")


def _match_conditions(msg):
    """Return diseases whose symptoms overlap the message (top 3 by overlap)."""
    if not DISEASE_SYMPTOMS:
        return []
    scores = []
    for disease, syms in DISEASE_SYMPTOMS.items():
        hits = sum(1 for s in syms if s.replace("_", " ") in msg)
        if hits:
            scores.append((hits, disease))
    scores.sort(reverse=True)
    return [d for _, d in scores[:3]]


@app.route("/chat", methods=["POST"])
@csrf.exempt
def chat():
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", ""))[:500]
    return jsonify(reply=bot_reply(message))


@app.route("/health")
def health():
    return jsonify(
        heart=heart_model is not None,
        obesity=obesity_model is not None,
        symptoms=symptoms_model is not None,
    )


init_db()

if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(debug=debug, port=5003)
