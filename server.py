from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import subprocess, json
from pathlib import Path

app = FastAPI()

ROOT = Path(__file__).resolve().parent
UPLOADS = ROOT / "uploads"
OUTDIR  = ROOT / "outputs" / "server_results"

UPLOADS.mkdir(parents=True, exist_ok=True)
OUTDIR.mkdir(parents=True, exist_ok=True)

@app.post("/upload_video")
async def upload_video(
    file: UploadFile = File(...),
    subject_id: str = Form(...),
    condition: str = Form(...),   # rest | exercise | breath
    modality: str = Form(...)     # face | palm
):
    # 1) save file
    dst = UPLOADS / file.filename
    with open(dst, "wb") as f:
        f.write(await file.read())

    # 2) call analyzer (IMPORTANT: run inside rppg_server_pack folder)
    try:
        proc = subprocess.run(
            [
                "python", str(ROOT / "realworld_demo_rppg_single.py"),
                "--video", str(dst),
                "--subject", subject_id,
                "--condition", condition,
                "--modality", modality,
                "--outdir", str(OUTDIR)
            ],
            capture_output=True, text=True, check=True
        )

        # stdout must be JSON
        data = json.loads(proc.stdout.strip())
        return JSONResponse(data)

    except subprocess.CalledProcessError as e:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "stderr": e.stderr, "stdout": e.stdout}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": str(e)}
        )
