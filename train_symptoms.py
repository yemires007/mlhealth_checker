"""
Train an illustrative symptom -> disease classifier for the ML Health app.

It builds a dataset from a small curated knowledge base (common conditions and
their typical symptoms), trains a Random Forest over a fixed symptom vocabulary,
and saves two files the app reads:

    models/symptoms_model.pkl      - the classifier
    models/symptoms_columns.json   - the ordered symptom vocabulary (feature order)

⚠️  ILLUSTRATIVE ONLY. The data is synthesised from the knowledge base below —
it is a working ML demo, NOT medical advice. To make it real, replace
DISEASE_SYMPTOMS with a labelled dataset (e.g. the public "Disease Symptom
Prediction" set: 132 symptoms / 41 diseases) and keep the same save format.

Run it from this folder, using the SAME scikit-learn version as the app
(see requirements.txt — 1.2.2), so the saved model loads in production:

    python train_symptoms.py
"""
import json
import random
from pathlib import Path

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

MODELS_DIR = Path(__file__).resolve().parent / "models"

# disease -> typical symptoms (symptom tokens use lowercase + underscores)
DISEASE_SYMPTOMS = {
    "Common Cold": ["runny_nose", "sneezing", "sore_throat", "cough",
                    "congestion", "mild_fever", "headache"],
    "Influenza": ["high_fever", "body_ache", "fatigue", "cough", "sore_throat",
                  "chills", "headache"],
    "COVID-19": ["fever", "dry_cough", "fatigue", "loss_of_smell",
                 "loss_of_taste", "shortness_of_breath", "sore_throat"],
    "Malaria": ["high_fever", "chills", "sweating", "headache", "nausea",
                "vomiting", "fatigue", "body_ache"],
    "Typhoid": ["prolonged_fever", "abdominal_pain", "weakness", "headache",
                "loss_of_appetite", "constipation", "fatigue"],
    "Migraine": ["headache", "nausea", "sensitivity_to_light", "blurred_vision",
                 "vomiting"],
    "Gastroenteritis": ["diarrhea", "vomiting", "abdominal_pain", "nausea",
                        "mild_fever", "dehydration"],
    "Pneumonia": ["cough", "high_fever", "shortness_of_breath", "chest_pain",
                  "fatigue", "chills"],
    "Allergy": ["sneezing", "itchy_eyes", "runny_nose", "rashes", "congestion"],
    "Asthma": ["shortness_of_breath", "wheezing", "chest_tightness", "cough"],
    "Dengue": ["high_fever", "headache", "joint_pain", "muscle_pain", "rashes",
               "nausea", "fatigue"],
    "Urinary Tract Infection": ["painful_urination", "frequent_urination",
                                "abdominal_pain", "mild_fever"],
}

SAMPLES_PER_DISEASE = 300
P_PRESENT = 0.85   # chance a typical symptom shows up for a case
P_NOISE = 0.03     # chance an unrelated symptom shows up


def build_dataset(vocab):
    rng = random.Random(42)
    idx = {s: i for i, s in enumerate(vocab)}
    X, y = [], []
    for disease, syms in DISEASE_SYMPTOMS.items():
        sset = set(syms)
        for _ in range(SAMPLES_PER_DISEASE):
            vec = [0] * len(vocab)
            for s in syms:
                if rng.random() < P_PRESENT:
                    vec[idx[s]] = 1
            for s in vocab:
                if s not in sset and rng.random() < P_NOISE:
                    vec[idx[s]] = 1
            if sum(vec) == 0:                       # never an all-zero row
                vec[idx[rng.choice(syms)]] = 1
            X.append(vec)
            y.append(disease)
    return np.array(X), np.array(y)


def main():
    np.random.seed(42)
    vocab = sorted({s for syms in DISEASE_SYMPTOMS.values() for s in syms})
    X, y = build_dataset(vocab)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, clf.predict(X_te))

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(clf, MODELS_DIR / "symptoms_model.pkl")
    (MODELS_DIR / "symptoms_columns.json").write_text(json.dumps(vocab, indent=2))

    print(f"Diseases: {len(DISEASE_SYMPTOMS)} | Symptoms: {len(vocab)} | "
          f"Samples: {len(X)}")
    print(f"Held-out accuracy: {acc:.1%}")
    print("Saved: models/symptoms_model.pkl, models/symptoms_columns.json")


if __name__ == "__main__":
    main()
