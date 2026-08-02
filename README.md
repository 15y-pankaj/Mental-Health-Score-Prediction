# 🧠 Mental Health Signal — Student Wellness Analytics

A small, end-to-end ML project that predicts a student's mental health score (0–10) from their social media habits, sleep, study patterns, and stress level, then serves the model through a FastAPI backend with a clean, single-page HTML frontend.

> ⚠️ **Disclaimer:** This is a personal/portfolio project built on a small synthetic-style dataset. It is **not** a clinical assessment tool and should never be used to diagnose, treat, or make decisions about anyone's mental health. If you're struggling, please talk to someone you trust or contact a professional helpline.

---

## ✨ What This Project Does

1. **Trains** a regression model (Random Forest + Optuna hyperparameter tuning) on 5,000 student records to predict a `Mental_Health_Score` from ~10 input features.
2. **Saves** the trained pipeline (preprocessing + model together) as a single `.joblib` file.
3. **Exposes** the model through a FastAPI endpoint with Pydantic input validation.
4. **Visualises** the prediction through a custom-built HTML/CSS/JS dashboard — a gauge chart, contextual band ("Signal: strained / balanced / strong"), and field-level error messages.

---

## 📁 Project Structure

```
mental_health_score_prediction/
├── app.py                       # FastAPI backend — load model + /predict endpoint
├── index.html                   # Single-page frontend
├── script.js                    # Frontend logic — form → fetch → render
├── style.css                    # Frontend styling
├── requirements.txt             # Pinned dependencies
├── README.md                    # This file
│
├── data/
│   └── Student Social Media And Mental Health Impact.csv
│
├── notebook/
│   ├── ML_Project.ipynb         # Training notebook (EDA → preprocessing → tuning)
│   └── catboost_info/           # CatBoost training logs (auto-generated)
│
├── model_pipeline/
│   └── rf_pipeline_tuned.joblib # Saved pipeline (preprocessor + model)
│
└── mhs_venv/                    # Local virtual environment (do not commit)
```

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.10+
- pip

### 2. Set up a virtual environment

```powershell
# Windows (PowerShell)
python -m venv mhs_venv
.\mhs_venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv mhs_venv
source mhs_venv/bin/activate
```

If PowerShell blocks the activation script, run once:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

This installs only what's needed to **run the app**. To also re-train the model from the notebook, uncomment the training-only section in `requirements.txt` first.

### 4. Make sure the trained model exists

Check that `model_pipeline/rf_pipeline_tuned.joblib` is present. If you don't have it:

1. Open `notebook/ML_Project.ipynb` in Jupyter
2. Run all cells in order — the final cell in section 12.3 saves the tuned pipeline automatically

### 5. Start the API

```powershell
uvicorn app:app --reload
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

### 6. Open the frontend

- **API docs (try /predict interactively):** http://127.0.0.1:8000/docs
- **Frontend UI:** open `index.html` directly OR serve it via VS Code's *Live Server* extension

The frontend expects the API at `http://127.0.0.1:8000` (configured in `script.js` line 4).

---

## 🧪 Testing the API

Send a request from any terminal:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 21,
    "gender": "Female",
    "country": "India",
    "academic_level": "Undergraduate",
    "most_used_platform": "Instagram",
    "purpose_of_use": "Entertainment",
    "avg_daily_usage_hours": 4.5,
    "daily_unlocks": 150,
    "study_hours": 3.0,
    "physical_activity_hours": 1.5,
    "sleep_hours_per_night": 7.0,
    "stress_level": "Medium"
  }'
```

Expected response:
```json
{"predicted_mental_health_score": 6.77}
```

---

## 📊 Model Performance

The tuned Random Forest pipeline is the production model.

| Metric | Training | Test |
|---|---|---|
| **R²**        | ~0.87  | ~0.83 |
| **MAE**       | —      | ~0.42 |

> Numbers vary slightly based on tuning runs. The tuning cell performs 100 Optuna trials with 5-fold cross-validation; the best parameters are then refit on the full training set.

For baseline comparison, the notebook also trains:
- **Linear Regression** (Test R² ≈ 0.74, MAE ≈ 0.54)
- **XGBoost** (tuned via Optuna)
- **CatBoost** (tuned via Optuna)

---

## 🔍 How It Works

### Backend (`app.py`)

A minimal FastAPI app that:

1. Loads the saved pipeline once at startup:
   ```python
   model = joblib.load(r"C:\Users\visha\MyProject\mental_health_score_prediction\model_pipeline\rf_pipeline_tuned.joblib")
   ```
2. Defines a Pydantic model (`StudentData`) that validates every incoming request — wrong types, out-of-range values, missing fields all return clean `422` errors.
3. Groups countries into the same top-10 + "Other" buckets that were used during training (so the model never sees a category it wasn't trained on).
4. Builds a one-row `pandas.DataFrame` matching the training column order and shape, then calls `model.predict(...)`.

### Frontend

A static HTML page served from `index.html`. Three pieces:

- **`index.html`** — semantic structure with three numbered groups (Profile, Academic & Digital Habits, Lifestyle & Stress) and a live result panel with four states (idle / loading / result / error).
- **`style.css`** — design tokens (colors, fonts, radii, shadows), responsive grid layout, dark gradient result panel, animated SVG gauge, segmented control for stress level.
- **`script.js`** — vanilla JS form handling, client-side validation that mirrors the Pydantic model, fetch to the API, and graceful error display.

No build step, no framework, no node modules. Open the HTML file and it works (with a local server for the API).

---

## 🧱 The Pipeline

The saved model is **not just the Random Forest** — it's the **entire preprocessing + model** wrapped in a scikit-learn `Pipeline`. This is important because:

- The input data needs a `log1p` transform on `Study_Hours`, standard scaling on numerics, ordinal encoding on `Stress_Level`, and one-hot encoding on the remaining categoricals.
- Replicating that by hand in production is fragile and error-prone.
- By saving the full pipeline, the API just calls `model.predict(raw_row)` and the right transformations are applied automatically.

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| ML | scikit-learn, NumPy, pandas |
| Tuning | Optuna (TPE sampler) |
| Baseline models | XGBoost, CatBoost (for comparison only) |
| Backend | FastAPI, Uvicorn, Pydantic |
| Frontend | Vanilla HTML, CSS, JavaScript (no framework) |
| Persistence | joblib |

---

## 📚 What I Learned Building This

- Why fitting all preprocessing inside a `Pipeline` (not outside) eliminates the risk of data leakage between train and inference.
- How to keep categorical cardinality under control (the `Country` column had 111 unique values — bucketed into top 10 + "Other").
- How cross-validation gives a more honest estimate of generalization than a single train/test split.
- How Optuna's TPE sampler finds better hyperparameters than random search in fewer trials.
- Why saving the **entire pipeline** (not just the model) is essential for clean deployment.
- How to design a FastAPI service with Pydantic validation that returns useful errors for both invalid inputs and server failures.

---

## ⚠️ Known Limitations

- **Small dataset (5,000 rows)** with synthetic-style features — the model can detect general patterns but won't reliably predict scores for individuals.
- **`scikit-learn==1.9.0` is pinned exactly** because models saved with newer scikit-learn versions don't always load with older ones. Don't change it without retraining.
- **Hardcoded model path** in `app.py` — fine for local use; for deployment, this should be an environment variable or a path relative to `__file__`.

---

## 📜 License

This project is for educational/portfolio purposes. The dataset was retrieved from a public source for analysis.

---

## 🙏 Acknowledgements

- Dataset: *Student Social Media And Mental Health Impact* (publicly available)
- Built as part of a structured learning project covering the full ML workflow — from raw data to a deployed, clickable demo.
