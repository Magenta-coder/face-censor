from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from concurrent.futures import ThreadPoolExecutor
import os, uuid, shutil, subprocess, shutil as _shutil, threading

from video_utils_dnn import censor_video_dnn

app = FastAPI(title="Video Face Censor (DNN SSD) – Pixelate Only + Progress")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True)
templates = Jinja2Templates(directory="templates")

os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

EXECUTOR = ThreadPoolExecutor(max_workers=2)

# PROGRESS store: job_id -> dict
PROGRESS = {}
LOCK = threading.Lock()

def has_ffmpeg(): return _shutil.which("ffmpeg") is not None

def _set_progress(job_id, current=None, total=None, status=None, out_path=None, error=None):
    with LOCK:
        info = PROGRESS.get(job_id, {})
        if current is not None: info["current"] = int(current)
        if total   is not None: info["total"] = int(total)
        if status  is not None: info["status"] = status
        if out_path is not None: info["out_path"] = out_path
        if error is not None: info["error"] = str(error)
        PROGRESS[job_id] = info

def _progress_cb_factory(job_id):
    def cb(done, total):
        _set_progress(job_id, current=done, total=total)
    return cb

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/blur-video-start")
async def api_blur_video_start(
    file: UploadFile = File(...),
    block_size: int = Form(18),
    pad_ratio: float = Form(0.25),
    rotate: int = Form(0),
    conf: float = Form(0.5),
    nms_iou: float = Form(0.45),
    use_flip: bool = Form(True),
    track_iou: float = Form(0.3),
    track_max_miss: int = Form(5),
    keep_audio: bool = Form(False)
):
    job_id = str(uuid.uuid4())
    in_ext = os.path.splitext(file.filename)[1].lower() or ".mp4"
    in_path = os.path.join("uploads", f"{job_id}{in_ext}")
    with open(in_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    out_path = os.path.join("outputs", f"{job_id}_censored.mp4")

    _set_progress(job_id, current=0, total=0, status="running", out_path=None, error=None)

    def job():
        try:
            censor_video_dnn(
                input_path=in_path, output_path=out_path,
                block_size=block_size, pad_ratio=pad_ratio,
                rotate=rotate, conf=conf,
                nms_iou=nms_iou, use_flip=use_flip,
                track_iou=track_iou, track_max_miss=track_max_miss,
                progress_fn=_progress_cb_factory(job_id)
            )
            final_path = out_path

            if keep_audio and has_ffmpeg():
                out_audio = os.path.join("outputs", f"{job_id}_censored_with_audio.mp4")
                cmd = [
                    "ffmpeg","-y","-i", out_path, "-i", in_path,
                    "-map","0:v:0","-map","1:a:0?","-c:v","copy","-c:a","aac","-shortest", out_audio
                ]
                try:
                    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    final_path = out_audio
                except subprocess.CalledProcessError:
                    final_path = out_path

            _set_progress(job_id, status="done", out_path=final_path)
        except Exception as e:
            _set_progress(job_id, status="error", error=str(e))

    EXECUTOR.submit(job)
    return {"ok": True, "job_id": job_id}

@app.get("/api/progress/{job_id}")
def api_progress(job_id: str):
    info = PROGRESS.get(job_id)
    if not info:
        return JSONResponse({"ok": False, "error": "job_id tidak ditemukan"}, status_code=404)
    current = info.get("current", 0)
    total = info.get("total", 0)
    percent = 0
    if total and total > 0:
        percent = int(round(100.0 * current / total))
        percent = max(0, min(100, percent))
    else:
        # jika total tidak diketahui: estimasi (naik perlahan)
        percent = min(95, current // 30)  # heuristik kecil agar UI tetap bergerak

    return {
        "ok": True,
        "status": info.get("status", "running"),
        "percent": percent,
        "current": current,
        "total": total,
        "ready": info.get("status") == "done",
        "error": info.get("error"),
    }

@app.get("/api/download/{job_id}")
def api_download(job_id: str):
    info = PROGRESS.get(job_id)
    if not info or info.get("status") != "done" or not info.get("out_path"):
        return JSONResponse({"ok": False, "error": "Belum siap atau job_id salah"}, status_code=400)
    path = info["out_path"]
    name = os.path.basename(path)
    return FileResponse(path, media_type="video/mp4", filename=name)