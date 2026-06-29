# Models go here

Place your three trained scikit-learn models in this folder, named **exactly**:

```
models/
├── heart_model.pkl
├── obesity_model.pkl
└── symptoms_model.pkl
```

The app loads these on startup with `joblib.load`. If a file is missing, the
app still runs — that prediction just returns a "model unavailable" message.
Check which models loaded at `/health`.

## Important

- **Same scikit-learn version.** The models must be trained/saved with the same
  scikit-learn as `requirements.txt` (1.2.2). A different version can fail to
  load or silently misbehave.
- **Commit the .pkl files** — Render deploys from Git, so the models must be in
  the repo (they are *not* git-ignored).
- **GitHub's file limit is 100 MB.** If any `.pkl` is larger, use
  [Git LFS](https://git-lfs.com) or host the model elsewhere and download it on
  startup. Random Forest models can be large — check with `ls -lh`.

## Feature formats the app expects

- **heart_model.pkl** — 15 features: `[age, gender, impulse, sys_bp, dia_bp,
  glucose, ckmb, troponin]` + 7 binary bins (computed in `app.py`).
- **obesity_model.pkl** — 5 features: `[age, gender, height, weight, bmi]`;
  returns a class label string.
- **symptoms_model.pkl** — currently `[age, gender, symptom_count]`. ⚠️ This is
  a placeholder encoding; adjust `predict_symptoms` in `app.py` to match however
  your symptoms model was actually trained (e.g. a fitted vectorizer over the
  symptom text).
