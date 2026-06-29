"""
ML Health — disease prediction web app.

A Flask app serving three scikit-learn models behind a simple UI:
  * Heart-disease risk (Random Forest)
  * Obesity classification
  * Disease-from-symptoms

Models are loaded from models/*.pkl. Loading is graceful: if a model file is
missing or fails to load, the app still starts and that prediction returns a
clear "model unavailable" message instead of crashing the whole service.

Run locally:
    python app.py        # http://127.0.0.1:5003
"""
import os
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

try:
    import joblib
except ImportError:  # joblib ships with scikit-learn; keep a clear message
    joblib = None

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

app = Flask(__name__)          # templates/ and static/ are the Flask defaults
CORS(app)


# --------------------------------------------------------------------------- #
# Model loading (graceful — a missing model never crashes the app)
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
# Symptoms needs its own model. (The original hearts_model.pkl is actually a
# 15-feature heart model, not a symptoms classifier, so it's not used here.)
symptoms_model = load_model("symptoms_model.pkl")


def model_unavailable(name):
    return jsonify(
        error=True,
        prediction=f"The {name} predictor isn't available in this demo yet.",
    ), 503


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
    return render_template("symptoms.html")


# --------------------------------------------------------------------------- #
# Heart disease
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
        pred = heart_model.predict(features)[0]
        return jsonify(prediction="Positive" if int(pred) == 1 else "Negative")
    except Exception as exc:  # noqa: BLE001
        return jsonify(error=True, prediction=f"Could not predict: {exc}"), 400


# --------------------------------------------------------------------------- #
# Obesity
# --------------------------------------------------------------------------- #
@app.route("/predict/obesity", methods=["POST"])
def predict_obesity():
    if obesity_model is None:
        return model_unavailable("obesity")
    try:
        d = request.get_json(force=True)
        age = float(d["age"]); gender = float(d["gender"])
        height = float(d["height"]); weight = float(d["weight"])
        bmi = weight / ((height / 100) ** 2)
        features = np.array([[age, gender, height, weight, bmi]])
        label = obesity_model.predict(features)[0]
        return jsonify(prediction=str(label))
    except ZeroDivisionError:
        return jsonify(error=True, prediction="Height must be greater than zero."), 400
    except Exception as exc:  # noqa: BLE001
        return jsonify(error=True, prediction=f"Could not predict: {exc}"), 400


# --------------------------------------------------------------------------- #
# Symptoms
# --------------------------------------------------------------------------- #
@app.route("/predict/symptoms", methods=["POST"])
def predict_symptoms():
    if symptoms_model is None:
        return model_unavailable("symptoms")
    try:
        d = request.get_json(force=True)
        age = float(d["age"]); gender = float(d["gender"])
        symptom_count = float(d.get("symptom_count", 0))
        # NOTE: must match the feature encoding used when the model was trained.
        features = np.array([[age, gender, symptom_count]])
        disease = symptoms_model.predict(features)[0]
        return jsonify(prediction=str(disease))
    except Exception as exc:  # noqa: BLE001
        return jsonify(error=True, prediction=f"Could not predict: {exc}"), 400


@app.route("/health")
def health():
    return jsonify(
        heart=heart_model is not None,
        obesity=obesity_model is not None,
        symptoms=symptoms_model is not None,
    )


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(debug=debug, port=5003)
