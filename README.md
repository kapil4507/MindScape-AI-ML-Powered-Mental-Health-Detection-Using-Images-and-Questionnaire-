# MindScape — Deployment Repository

This repository contains the deployment artifact for the MindScape web application (the service that runs the UI and serves the prediction endpoint). Model training and model/prediction utilities live in a separate repository: [MindScape-Training-and-Prediction](https://github.com/kapil4507/MindScape-Training-and-Prediction). Please go there to train models and to find model artifact exports, training scripts, and prediction utilities.

This README documents everything required to deploy this project (locally and in production) using model artifacts and assets produced from the training repository.

---

## Contents (what this repo has)
- `app.py` — Flask application that exposes web routes and the `/predict` endpoint. Implements:
  - Image preprocessing and inference helper functions (emotion, posture, visibility / occlusion).
  - A fusion engine that aggregates per-image and questionnaire signals into a final report object.
  - Templates rendering for UI and results.
- `templates/`
  - `index.html` — main frontend UI (React UMD + Tailwind via CDN). Handles auth placeholders, questionnaire (GAD‑7 / PHQ‑9), image upload, consent, and form submission.
  - `result.html` — server-side result rendering.
- `static/` (optional) — place static assets if present/used.
- `models/` (not checked in) — directory where trained model artifact files must be placed for inference.
- `class_names.txt` (expected by the code) — label names used to interpret model outputs.

---

## Quick overview of the runtime behavior
- The frontend (served from `/`) shows the UI, collects questionnaire answers, and uploads images (form POST to `/predict`).
- The backend receives the multipart form data at `/predict`, runs per-image analyses, aggregates scores and questionnaire totals, runs the fusion engine, and returns a result page showing:
  - Final wellness score and persona label
  - AI visual analysis breakdown (facial expression, body language, visibility)
  - Questionnaire summary
- The `/predict` endpoint expects:
  - `files[]` — one or more image files (the app UI enforces exactly 4 images)
  - `gad_score` and `phq_score` — integers submitted as form fields

---

## Prerequisites

- Python 3.8+ (3.10 recommended)
- pip
- git
- (Optional) virtualenv or venv for isolation
- Model artifacts and label files exported from the training repo [MindScape-Training-and-Prediction](https://github.com/kapil4507/MindScape-Training-and-Prediction)

---

## Install & run locally

1. Clone this repository
```bash
git clone https://github.com/kapil4507/MindScape-AI-ML-Powered-Mental-Health-Detection-Using-Images-and-Questionnaire-.git
```

2. Create and activate a virtual environment (optional but recommended)
```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

3. Install Python dependencies
- If there is a `requirements.txt` file in the repo:
```bash
pip install -r requirements.txt
```
- Example dependency list you can use if a requirements file is not present:
```bash
pip install flask numpy opencv-python tensorflow pillow python-dotenv
```

4. Prepare model artifacts and labels
- Download or copy trained model files and label files from the training repository:
  - See [MindScape-Training-and-Prediction](https://github.com/kapil4507/MindScape-Training-and-Prediction) for training/export instructions and model artifacts.
- Place model files into the `models/` directory at the project root (create the directory if it does not exist).
- Place the label file `class_names.txt` (if used by the app) in the project root (or adjust `app.py` to point to the correct path).

5. Environment configuration
- Copy and edit a `.env` file or export environment variables used by the app:
  - Example `.env`:
    ```env
    FLASK_ENV=production
    FLASK_RUN_PORT=5000
    SECRET_KEY=replace-with-your-secret
    UPLOAD_FOLDER=uploads
    MAX_CONTENT_LENGTH=16*1024*1024
    ```
  - The frontend uses Firebase placeholders inside `templates/index.html`. If you want client-side authentication/storage enabled, update the Firebase configuration inside that template or provide a mechanism to inject it.

6. Create upload directory
```bash
mkdir -p uploads
```

7. Start the Flask app (simple run)
```bash
python app.py
```
- By default the app listens on port 5000. Visit http://127.0.0.1:5000/ to view the UI.

8. Test the `/predict` route with curl (example)
- This example sends two integer fields and four files named a.jpg..d.jpg:
```bash
curl -X POST -F "gad_score=3" -F "phq_score=5" \
  -F "files[]=@a.jpg" -F "files[]=@b.jpg" -F "files[]=@c.jpg" -F "files[]=@d.jpg" \
  http://127.0.0.1:5000/predict
```

---

## Production deployment recommendations
- Run under a production WSGI server (Gunicorn / uWSGI)
  - Example using gunicorn:
    ```bash
    pip install gunicorn
    gunicorn -w 4 -b 0.0.0.0:8000 app:app
    ```
- Serve behind a reverse proxy (Nginx) to handle TLS/HTTPS and static assets.
- Limit upload sizes and sanitize inputs. Configure `MAX_CONTENT_LENGTH` and validate file types in `app.py`.
- If you use persistent storage for uploaded images or results, ensure secure storage, access controls, and an appropriate retention policy.
- Consider using Docker for reproducible deployments (Dockerfile + docker-compose). Example Dockerfile snippet:
  ```dockerfile
  FROM python:3.10-slim
  WORKDIR /app
  COPY . .
  RUN pip install --no-cache-dir -r requirements.txt
  ENV FLASK_ENV=production
  EXPOSE 5000
  CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
  ```

---

## Configuration references (code-level)
- `app.py` contains settings and constants that can be adjusted:
  - `UPLOAD_FOLDER` — where uploaded images are saved
  - Model load paths (edit to match your artifact filenames and locations)
  - Fusion engine weights (if you want to change weighting)
- `templates/index.html` contains the client-side Firebase config placeholders. Replace those with your actual Firebase project values if you want frontend auth/firestore functionality.

---

## API / Endpoint summary
- GET `/` — serves the main UI
- POST `/predict` — accepts multipart form submission:
  - Form fields:
    - `gad_score` (integer)
    - `phq_score` (integer)
  - File fields:
    - `files[]` (multiple image uploads; frontend enforces 4 images)
  - Returns a rendered HTML result page containing the fused score and analysis.

---

## Troubleshooting
- App prints model load and inference debug messages to console — check logs for stack traces.
- If the app cannot find or load a model, check model file paths in `app.py`.
- If images fail to process, ensure `opencv-python` is installed and that Python can open the image files.
- For performance issues, consider running TensorFlow with GPU support or pre-warming models at startup.


---

## Where to get training & model artifacts
- Training code, prediction/training utilities, datasets and model export instructions are available in:
  - [MindScape-Training-and-Prediction](https://github.com/kapil4507/MindScape-Training-and-Prediction)
- Use that repository to train and export artifacts; then copy the exported models and any label files into this deployment repository before running the server.

---

## Folder structure (example)
- app.py
- templates/
  - index.html
  - result.html
- models/                 (place exported model files here)
- class_names.txt         (place label names here if used)
- uploads/                (runtime upload directory)
- requirements.txt
- .env (optional)

