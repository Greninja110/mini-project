# Real-time Object Detection with 3D AR

A browser-based real-time object detection system that streams live video from an Android phone (via IP Webcam), runs YOLOv8 inference on an NVIDIA GPU, and overlays detection results and interactive 3D AR models directly in the browser — all served from a single Flask application.

The system is designed for local network use: your phone acts as a wireless camera, your PC runs the model, and any browser on the same network can watch the annotated live feed. The AR mode lets you place 3D furniture models (chair, table, bed) directly on top of the video frame using Three.js.

The pipeline is fully GPU-accelerated using PyTorch CUDA with FP16 (half-precision) and TF32 optimizations, achieving real-time inference speeds on hardware as modest as an RTX 3050 Laptop GPU. All five YOLOv8 model variants (nano through extra-large) are bundled, making it easy to trade accuracy for speed.

> **Note:** The 3D AR feature is currently a work in progress and not fully implemented. Basic model placement via Three.js is functional, but real depth estimation, accurate surface anchoring, and complete GLB model support are still under development.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.x |
| Web Framework | Flask 2.3.3 |
| Object Detection | Ultralytics YOLOv8 (ultralytics 8.1.8) |
| Deep Learning Backend | PyTorch 2.2.0 + CUDA 12.1 |
| Computer Vision | OpenCV 4.8.1 |
| 3D AR Rendering | Three.js r132 + GLTFLoader |
| System Monitoring | psutil 5.9.5 |
| 3D Model Format | GLTF/GLB |
| Streaming Protocol | MJPEG (multipart HTTP) |

---

## Detailed Description

### Complete Flow

```
Android Phone (IP Webcam app)
        │  MJPEG stream over LAN
        ▼
frame_capture_thread  ──► frame_queue (maxsize=1, drops stale frames)
        │
        ▼
process_frames_thread
  ├─ YOLOv8 inference (GPU, FP16, CUDA AMP)
  ├─ Annotates frame with bounding boxes + labels
  ├─ Writes to global: processed_frame
  └─ Writes to global: detected_objects [ {name, confidence, bbox} ]
        │
        ▼
generate_frames()  ──► JPEG encode ──► multipart HTTP stream
        │
        ▼
Browser  ◄──── /video_feed  (live annotated stream)
         ◄──── /stats        (FPS, model name, status)  — polled every 2s
         ◄──── /detected_objects  (list of objects in current frame)
         ◄──── /system_stats      (CPU %, RAM, GPU %, GPU VRAM)

update_system_stats  (separate thread, updates every 1s)
```

The `frame_queue` is intentionally capped at size 1. When the queue is full, the oldest frame is discarded before inserting the new one — this keeps latency minimal by always feeding the model the most recent frame rather than processing a backlog.

### Object Detection

- **Models available:** `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, `yolov8x.pt` — all five are in the repo; switch by changing `MODEL_PATH` in `app.py`.
- **GPU path:** Model is moved to `cuda:0`, converted to FP16 (`model.model.half()`), and inference runs under `torch.cuda.amp.autocast`. `cudnn.benchmark = True` is set for additional speed.
- **CPU fallback:** If CUDA is unavailable the model loads in FP32 on CPU automatically.
- **Confidence threshold:** 0.5 by default (`CONFIDENCE_THRESHOLD`).
- **Frame skip:** `PROCESS_EVERY_N_FRAMES = 1` means every frame is processed; increase to 2 or 3 to reduce GPU load on slower hardware.
- **Resize factor:** Frames are scaled to 90% of their original resolution before inference to reduce compute without visually degrading the stream.

### Browser UI Features

- **Live annotated video feed** — YOLOv8 bounding boxes and class labels rendered directly on the MJPEG stream image element.
- **FPS + GPU memory overlay** — drawn onto the frame server-side using OpenCV `putText`.
- **Detected Objects Panel** — togglable sidebar listing every detected class and its confidence score. Clicking any object opens a Google search for it in a new tab.
- **System Statistics Panel** — live progress bars for CPU usage, RAM, integrated GPU (approximated from CPU), NVIDIA GPU (VRAM-based), and GPU memory with human-readable byte formatting.
- **Fullscreen mode** — fullscreen the video container via the Fullscreen API.
- **AR Mode (Three.js overlay)** — an `<canvas>` sits transparently above the video feed. Selecting a furniture item (chair / table / bed) and clicking the video spawns the corresponding `.glb` model at the clicked 3D world position. Uses a `PerspectiveCamera` ray-unprojection to map screen clicks to scene coordinates. A simulated "analyzing environment" progress bar (3 s) precedes placement.

### Logging

Every run creates a timestamped log file at `logs/yolo_detection_YYYYMMDD_HHMMSS.log` (also mirrored to stdout). Logs cover CUDA compatibility checks, model loading, GPU verification, thread startup, stream errors, and system stat exceptions.

### Key Configuration (top of `app.py`)

| Constant | Default | What to change it for |
|---|---|---|
| `IP_WEBCAM_URL` | `http://10.122.72.173:8081` | Your phone's IP Webcam address |
| `MODEL_PATH` | `yolov8n.pt` | Swap to `yolov8m.pt` etc. for higher accuracy |
| `CONFIDENCE_THRESHOLD` | `0.5` | Lower for more detections, raise to filter noise |
| `PROCESS_EVERY_N_FRAMES` | `1` | Increase to skip frames and reduce GPU load |
| `RESIZE_FACTOR` | `0.90` | Lower (e.g. `0.5`) for faster inference |
| `JPEG_QUALITY` | `90` | Lower to reduce stream bandwidth |

---

## Setup & Run

### Prerequisites

- Python 3.9+
- NVIDIA GPU with CUDA 12.1 drivers (`nvidia-smi` must work)
- Android phone with [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam) app running on the same LAN

### Install

```bash
# 1. Install PyTorch with CUDA support first (must be done separately)
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

# 2. Install remaining dependencies
pip install -r requirements.txt
```

On Windows, run `setup_env.bat` to verify your NVIDIA driver and PyTorch CUDA setup before proceeding.

### Run

```bash
# Update IP_WEBCAM_URL in app.py to match your phone's IP, then:
python app.py
```

Open `http://localhost:5000` in your browser (or replace `localhost` with the server's LAN IP to watch from another device).

### AR Models

Place `.glb` format 3D models in `static/ar/`:

```
static/ar/chair.glb   ✅ included
static/ar/table.glb   ❌ missing — add manually
static/ar/bed.glb     ❌ missing — add manually
```

Free `.glb` models can be found on [Sketchfab](https://sketchfab.com) or [Google Poly](https://poly.google.com).

---

## Future Expansion

### Near-term improvements

- **Fix AR `loadModels()` bug** — a dangling `modelLoader.load()` call outside the `forEach` loop (line ~876 in `index.html`) references `modelName` out of scope, causing a ReferenceError in the console. The `forEach` block already handles loading correctly; the extra call should be removed.
- **Add missing GLB models** — `table.glb` and `bed.glb` need to be sourced and placed in `static/ar/`.
- **Config file / `.env` support** — move `IP_WEBCAM_URL` and other tunable constants out of source code into an environment file so the app can run without code edits.
- **Production WSGI deployment** — swap Flask's built-in dev server for Gunicorn + a reverse proxy (nginx) for multi-client stability.

### Feature expansions

- **Real depth estimation for AR** — replace the simulated 3-second "analyzing" progress bar with an actual monocular depth model (e.g. MiDaS) to compute real surface normals and place AR objects accurately on floors/tables rather than using simplified ray-unprojection.
- **Object tracking** — integrate SORT or ByteTrack so bounding boxes persist across frames with stable IDs instead of being re-detected independently each frame. This would also enable trajectory visualization.
- **Detection alerts / notifications** — trigger a browser notification or webhook when a specific object class (e.g. "person", "car") enters the frame. Useful for surveillance use cases.
- **Detection history & database logging** — persist `detected_objects` to SQLite or PostgreSQL with timestamps, enabling playback, search, and heatmap generation of where objects appeared over time.
- **Screenshot / clip recording** — add a server-side endpoint to save the current annotated frame or record a rolling buffer of the last N seconds to disk.
- **WebRTC streaming** — replace the MJPEG `<img>` stream with a WebRTC peer connection for sub-100ms latency instead of the current ~300–500ms MJPEG overhead.
- **Multiple camera support** — extend the capture/process thread architecture to handle multiple simultaneous IP Webcam streams, each with their own queue and model instance.
- **Custom model fine-tuning** — add a `/train` endpoint or companion script to fine-tune YOLOv8 on user-provided images for domain-specific detection (e.g. industrial parts, specific products).
- **Mobile-responsive AR** — extend AR support to work on mobile browsers using WebXR Device API for true phone-based augmented reality.
- **Model hot-swap** — expose a `/set_model` endpoint to switch between YOLOv8 variants at runtime without restarting the server, so users can compare speed vs. accuracy live.

---

## Project Structure

```
real_mini_project/
├── app.py                  # Entire backend: Flask app + 3 daemon threads
├── requirements.txt        # Python dependencies (PyTorch installed separately)
├── setup_env.bat           # Windows CUDA/driver verification script
├── yolov8n.pt              # YOLOv8 Nano   — fastest, least accurate
├── yolov8s.pt              # YOLOv8 Small
├── yolov8m.pt              # YOLOv8 Medium
├── yolov8l.pt              # YOLOv8 Large
├── yolov8x.pt              # YOLOv8 XLarge — slowest, most accurate
├── templates/
│   └── index.html          # Full UI: video feed, AR canvas, panels, Three.js
├── static/
│   └── ar/
│       └── chair.glb       # 3D chair model (table.glb, bed.glb needed)
└── logs/                   # Auto-created; one log file per run
```

---

## License

This project is for educational and personal use.
