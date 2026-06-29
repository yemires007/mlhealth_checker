# ML Health — disease prediction web app

A Flask app serving three scikit-learn models behind a simple multi-page UI:

- **Heart-disease risk** (Random Forest) — from vitals and blood markers.
- **Obesity classification** — from age, gender, height, weight (BMI computed).
- **Disease from symptoms** — multi-hot symptom vector → likely condition
  (illustrative model; see `train_symptoms.py`).

Models are loaded from `models/*.pkl`. Loading is **graceful**: a missing model
doesn't crash the app — that prediction returns a clear "model unavailable"
message, and you can see what loaded at `/health`.

It also has **user accounts** (sign up / log in), a **saved health-check
history**, a **dashboard**, a **health-analysis** page, and a rule-based
**health assistant chatbot** (healthy ranges, symptom hints, which-check
guidance, with safety disclaimers). Predictions are saved to your history when
you're logged in.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
# put your models in models/  (see models/README.md)
python app.py
```

Open **http://127.0.0.1:5003**.

## What changed from the original
- One clean folder: `templates/` + `static/` (Flask defaults) instead of
  `../frontend` / `../static` relative paths.
- Models load from a fixed `models/` path (not the current working directory).
- **Graceful model loading** — the app starts even if a `.pkl` is missing.
- Per-request error handling — bad input or a model error returns a JSON message
  instead of a 500.
- Fixed the symptoms page's gender bug (the form never submitted before).

## The symptoms model
`train_symptoms.py` builds an **illustrative** symptom→disease classifier from a
small curated knowledge base, saving `models/symptoms_model.pkl` and
`models/symptoms_columns.json` (the symptom vocabulary the app vectorises
against). It's a working ML demo, not medical advice — swap in a real labelled
dataset to make it authoritative. Retrain with:
```bash
python train_symptoms.py
```

## Files
```
diseaseprediction/
├── app.py                  # routes + model loading + prediction
├── train_symptoms.py       # trains the illustrative symptoms model
├── templates/              # index, heart, obesity, symptoms
├── static/styles.css
├── models/                 # heart_model.pkl, obesity_model.pkl,
│                           # symptoms_model.pkl, symptoms_columns.json
├── requirements.txt
├── render.yaml, Procfile   # Render deploy config
└── README.md
```

## Deploying (Render)
This fits Render's free tier (scikit-learn + numpy, unlike deep-learning
frameworks). Push to GitHub, then **New + → Blueprint** on Render — it reads
`render.yaml`. See the bank/expense apps' guides for the exact clicks.

> The model `.pkl` files must be committed to the repo (they're not
> git-ignored). If any file exceeds GitHub's 100 MB limit, use Git LFS.
