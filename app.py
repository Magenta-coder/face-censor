from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import os, uuid, shutil, subprocess, shutil as _shutil

from video_utils_dnn import censor_video_dnn

app = FastAPI(title="Video Face Censor (DNN SSD)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)
templates = Jinja2Templates(directory="templates")

os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

def has_ffmpeg():
    return _shutil.which("ffmpeg") is not None

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/blur-video")
async def api_blur_video(
    file: UploadFile = File(...),
    mode: str = Form("blur"),           # "blur" atau "pixelate"
    strength: int = Form(31),
    padding: int = Form(20),
    rotate: int = Form(0),              # 0/90/180/270
    conf: float = Form(0.5),            # 0.1..0.9
    keep_audio: bool = Form(False)      # gabungkan audio asli via ffmpeg
):
    # Simpan file input ke uploads/
    in_id = str(uuid.uuid4())
    in_ext = os.path.splitext(file.filename)[1].lower() or ".mp4"
    in_path = os.path.join("uploads", f"{in_id}{in_ext}")
    with open(in_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Proses ke outputs/
    out_path = os.path.join("outputs", f"{in_id}_censored.mp4")
    try:
        censor_video_dnn(
            input_path=in_path, output_path=out_path,
            mode=mode, strength=strength, padding=padding,
            rotate=rotate, conf=conf
        )
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    # Merge audio (opsional)
    if keep_audio and has_ffmpeg():
        out_audio = os.path.join("outputs", f"{in_id}_censored_with_audio.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-i", out_path,
            "-i", in_path,
            "-map", "0:v:0", "-map", "1:a:0?",
            "-c:v", "copy", "-c:a", "aac", "-shortest",
            out_audio
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return FileResponse(out_audio, media_type="video/mp4", filename=os.path.basename(out_audio))
        except subprocess.CalledProcessError:
            # jika merge gagal, kirim versi tanpa audio
            return FileResponse(out_path, media_type="video/mp4", filename=os.path.basename(out_path))

    return FileResponse(out_path, media_type="video/mp4", filename=os.path.basename(out_path))
