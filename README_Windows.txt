# RealWorld rPPG Server – Windows Setup Guide

This guide explains how to run the FastAPI-based rPPG server on Windows.
It assumes you already have Python 3.9+ installed.

---

## 1. Create and Activate Virtual Environment
Open PowerShell inside your project folder and run:

    python -m venv .venv
    .\.venv\Scripts\activate

---

## 2. Install Dependencies
Make sure you have the requirements.txt file in the same folder.
Then run:

    pip install -r requirements.txt

---

## 3. Start the rPPG Server
Run the FastAPI server with uvicorn:

    uvicorn server:app --reload --host 0.0.0.0 --port 8000

- `server:app` → means FastAPI will look for `app = FastAPI()` inside `server.py`
- `--reload`   → auto-restarts when you change code
- `--host 0.0.0.0` → makes it accessible on your network
- `--port 8000` → server runs at http://127.0.0.1:8000

---

## 4. Test the Server
Open your browser and go to:

    http://127.0.0.1:8000

You should see a welcome message like:  
`{"message": "rPPG server is running"}`

---

## 5. API Documentation
FastAPI automatically generates interactive docs:

Swagger UI → http://127.0.0.1:8000/docs  
ReDoc      → http://127.0.0.1:8000/redoc

Here you can upload a video file (`.mp4`) and get estimated HR, HRV, RR, and SpO₂ trend index.

---

## 6. Stop the Server
Press `CTRL + C` in PowerShell to stop it.

---

# Notes
- Make sure your videos are small test clips first (face or palm, ~10–30s).
- Results will be saved into a folder like `outputs/`.
- You can extend this to a **mobile app** later by calling the same API from a phone app.

