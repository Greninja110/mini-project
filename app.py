import cv2
import torch
import numpy as np
from flask import Flask, render_template, Response, jsonify, send_from_directory
import time
import threading
import queue
import os
import psutil
import platform
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor
import logging
import datetime

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Set up logging
log_filename = f"logs/yolo_detection_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Create static folders if they don't exist
os.makedirs('static/ar', exist_ok=True)

# Configuration
IP_WEBCAM_URL = "http://192.168.5.73:8080/video"  # Your IP Webcam URL
MODEL_PATH = "yolov8n.pt"  # Using YOLOv8 nano for speed
CONFIDENCE_THRESHOLD = 0.5
USE_GPU = True
PROCESS_EVERY_N_FRAMES = 1  # Process every 2nd frame to reduce load
RESIZE_FACTOR = 0.90  # Resize factor for input frames (0.75 = 75% of original size)
MAX_QUEUE_SIZE = 1  # Maximum queue size for frames
JPEG_QUALITY = 90  # JPEG quality for streaming (lower = faster but lower quality)

# Global variables
model = None
frame_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
processed_frame = None
processing_active = True
current_fps = 0
detected_objects = []  # Store detected objects

# System stats variables
system_stats = {
    "cpu_percent": 0,
    "memory_used": 0,
    "memory_total": 0,
    "integrated_gpu_percent": 0,
    "nvidia_gpu_percent": 0,
    "gpu_memory_used": 0,
    "gpu_memory_total": 0
}

def check_cuda_compatibility():
    """Check if CUDA version is compatible with PyTorch version"""
    logger.info("Checking CUDA and PyTorch compatibility...")
    
    if not torch.cuda.is_available():
        logger.error("CUDA is not available!")
        return False
    
    cuda_version = torch.version.cuda
    pytorch_version = torch.__version__
    
    logger.info(f"CUDA version: {cuda_version}")
    logger.info(f"PyTorch version: {pytorch_version}")
    
    # Check compatibility (simplified check)
    # For most accurate compatibility, check PyTorch documentation
    cuda_major = int(cuda_version.split('.')[0])
    
    if cuda_major > 12:
        logger.warning("Your CUDA version is newer than PyTorch's officially supported versions. This might cause issues.")
    elif cuda_major < 11:
        logger.warning("Your CUDA version might be too old for your PyTorch version.")
    
    return True

def verify_gpu_usage():
    """Verify GPU is detected and being used correctly"""
    logger.info("\n--- GPU VERIFICATION ---")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"Current device: {torch.cuda.current_device()}")
        logger.info(f"Device name: {torch.cuda.get_device_name(0)}")
        logger.info(f"Device count: {torch.cuda.device_count()}")
        logger.info(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        logger.info(f"Memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    else:
        logger.info("CUDA is not available. Using CPU only.")
    logger.info("------------------------\n")

def initialize_model():
    """Initialize the YOLOv8 model with proper GPU configuration"""
    global model
    
    # First check for CUDA availability
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available! Using CPU instead.")
        model = YOLO(MODEL_PATH)
        return
    
    # CUDA is available, configure it properly
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Set device explicitly
    device = torch.device("cuda:0")
    
    # Enable CUDA optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True
    
    try:
        # Enable TF32 optimization if available
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("TF32 optimization enabled")
    except:
        logger.info("TF32 optimization not available")
    
    # Load model and explicitly move to GPU
    logger.info(f"Loading model {MODEL_PATH} to GPU...")
    model = YOLO(MODEL_PATH)
    model.to(device)
    
    # Enable half-precision for faster inference
    logger.info("Enabling half precision (FP16)")
    model.model.half()
    
    # Force a small GPU operation to verify CUDA is working
    logger.info("Running test inference to verify GPU...")
    try:
        dummy_input = torch.rand(1, 3, 320, 320, device=device)
        with torch.no_grad():
            _ = model.model(dummy_input.half())
        
        logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
        logger.info(f"GPU initialization successful!")
    except Exception as e:
        logger.error(f"Error during test inference: {e}")
        logger.error("GPU initialization failed, check CUDA setup")

def frame_capture_thread():
    """Thread to capture frames from IP webcam"""
    global processing_active
    
    logger.info("Starting frame capture thread...")
    
    cap = cv2.VideoCapture(IP_WEBCAM_URL)
    if not cap.isOpened():
        logger.error("Error: Could not open video stream.")
        return
    
    # Set optimal buffer size to reduce latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Try to set the lowest possible latency mode
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    
    frame_count = 0
    
    while processing_active:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to receive frame.")
            time.sleep(0.1)  # Prevent CPU spinning
            continue
        
        frame_count += 1
        
        # Process only every Nth frame
        if frame_count % PROCESS_EVERY_N_FRAMES != 0:
            continue
        
        # Resize frame to reduce processing time if needed
        if RESIZE_FACTOR != 1.0:
            width = int(frame.shape[1] * RESIZE_FACTOR)
            height = int(frame.shape[0] * RESIZE_FACTOR)
            frame = cv2.resize(frame, (width, height))
        
        # Drop frames if queue is full (keep only the most recent frame)
        if frame_queue.full():
            try:
                frame_queue.get_nowait()  # Remove oldest frame
            except queue.Empty:
                pass
        
        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            pass
    
    cap.release()
    logger.info("Frame capture thread stopped")

def process_frames_thread():
    """Thread to process captured frames using YOLOv8"""
    global processed_frame, current_fps, processing_active, detected_objects
    
    logger.info("Starting frame processing thread...")
    
    frame_times = []
    max_times = 10  # Number of frames to average FPS over
    
    # Explicitly set CUDA device if available
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        logger.info(f"Processing frames on {torch.cuda.get_device_name(0)}")
    
    while processing_active:
        try:
            # Get frame from queue with a timeout
            frame = frame_queue.get(timeout=1.0)
            
            start_time = time.time()
            
            # Skip processing if model isn't loaded yet
            if model is None:
                processed_frame = frame
                continue
            
            # Run YOLOv8 inference with optimized settings
            if torch.cuda.is_available():
                # Explicitly ensure processing on GPU
                with torch.cuda.amp.autocast(enabled=True):
                    results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False, device=0)
            else:
                results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
            
            # Update detected objects list
            current_objects = []
            for r in results[0].boxes.data.tolist():
                x1, y1, x2, y2, confidence, class_id = r
                class_name = results[0].names[int(class_id)]
                current_objects.append({
                    "name": class_name,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2]
                })
            
            # Update global detected objects
            detected_objects = current_objects
            
            # Visualize results
            processed_frame = results[0].plot()
            
            # Calculate processing time and FPS
            processing_time = time.time() - start_time
            frame_times.append(processing_time)
            
            # Keep only the most recent times for averaging
            if len(frame_times) > max_times:
                frame_times.pop(0)
            
            # Calculate average FPS
            avg_time = sum(frame_times) / len(frame_times)
            current_fps = 1.0 / avg_time if avg_time > 0 else 0
            
            # Add FPS counter to the frame
            cv2.putText(processed_frame, f"FPS: {current_fps:.1f}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)  # Black text with thick outline
            cv2.putText(processed_frame, f"FPS: {current_fps:.1f}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 190), 1)  # Green text
            
            # Add GPU info to frame if available
            if torch.cuda.is_available():
                gpu_usage = torch.cuda.memory_allocated(0) / (1024**3)  # GB
                cv2.putText(processed_frame, f"GPU: {gpu_usage:.2f}GB", (20, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
                cv2.putText(processed_frame, f"GPU: {gpu_usage:.2f}GB", (20, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 190), 1)
            
        except queue.Empty:
            # No frame available, just continue
            continue
        except Exception as e:
            logger.error(f"Error in processing thread: {e}")
            time.sleep(0.1)  # Prevent CPU spinning

def generate_frames():
    """Generate frames for the Flask video feed"""
    global processed_frame
    
    while True:
        # Wait until a processed frame is available
        if processed_frame is not None:
            # Convert to JPEG for streaming with optimized quality
            ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()
            
            # Yield the frame in the format required by Flask
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Small sleep to prevent the function from hogging CPU
        time.sleep(0.01)

def update_system_stats():
    """Update system statistics including GPU usage"""
    global system_stats
    
    logger.info("Starting system stats monitoring thread...")
    
    while True:
        try:
            # Get CPU usage
            system_stats["cpu_percent"] = psutil.cpu_percent()
            
            # Get memory usage
            memory = psutil.virtual_memory()
            system_stats["memory_used"] = memory.used
            system_stats["memory_total"] = memory.total
            
            # Update NVIDIA GPU stats if available
            if torch.cuda.is_available():
                try:
                    system_stats["nvidia_gpu_percent"] = 0  # Start with default value
                    
                    # Get GPU usage percent - this is approximate as PyTorch doesn't provide direct usage %
                    # We can infer it from memory usage
                    gpu_memory_total = 0
                    if hasattr(torch.cuda, 'get_device_properties'):
                        props = torch.cuda.get_device_properties(0)
                        if hasattr(props, 'total_memory'):
                            gpu_memory_total = props.total_memory
                    
                    gpu_memory_used = torch.cuda.memory_allocated(0)
                    gpu_memory_reserved = torch.cuda.memory_reserved(0)
                    
                    # Calculate usage percentage based on allocated vs total
                    if gpu_memory_total > 0:
                        nvidia_gpu_percent = (gpu_memory_used / gpu_memory_total) * 100
                        system_stats["nvidia_gpu_percent"] = nvidia_gpu_percent
                    
                    system_stats["gpu_memory_used"] = gpu_memory_used
                    system_stats["gpu_memory_total"] = gpu_memory_total or (4 * 1024 * 1024 * 1024)  # 4GB default for RTX 3050
                except Exception as e:
                    logger.error(f"Error getting GPU stats: {e}")
            
            # Simulated integrated GPU usage - just an estimate based on CPU
            system_stats["integrated_gpu_percent"] = min(system_stats["cpu_percent"] * 1.2, 100)
            
        except Exception as e:
            logger.error(f"Error updating system stats: {e}")
        
        # Update every second
        time.sleep(1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/static/ar/<path:path>')
def serve_ar_assets(path):
    return send_from_directory('static/ar', path)

@app.route('/stats')
def stats():
    stats_data = {
        "status": "Online",
        "model": MODEL_PATH,
        "fps": f"{current_fps:.1f}",
    }
    
    # Add GPU info if available
    if torch.cuda.is_available():
        try:
            mem_allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)  # Convert to GB
            mem_reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
            stats_data["gpu_mem"] = f"{mem_allocated:.2f}GB / {mem_reserved:.2f}GB"
            stats_data["gpu_name"] = torch.cuda.get_device_name(0)
        except:
            pass
    
    return jsonify(stats_data)

@app.route('/detected_objects')
def get_detected_objects():
    return jsonify(detected_objects)

@app.route('/system_stats')
def system_stats_endpoint():
    # Return the globally maintained system stats
    return jsonify(system_stats)

def setup_ar_assets():
    """Set up directory for AR 3D models"""
    ar_dir = 'static/ar'
    os.makedirs(ar_dir, exist_ok=True)
    
    # Check for 3D model files
    required_models = ['chair.glb', 'table.glb', 'bed.glb']
    
    logger.info("Checking for required 3D models...")
    missing_models = []
    for model_file in required_models:
        model_path = os.path.join(ar_dir, model_file)
        if not os.path.exists(model_path):
            missing_models.append(model_file)
    
    if missing_models:
        logger.warning(f"Warning: The following model files are missing from {ar_dir}:")
        for model in missing_models:
            logger.warning(f"  - {model}")
        logger.warning(f"\nPlease place your 3D models (.glb format) in the {ar_dir} directory.")
    else:
        logger.info("All required 3D models found!")
    
    logger.info(f"AR setup complete. 3D models should be in {ar_dir} directory.")

def start_threads():
    """Start all background threads for the application"""
    logger.info("Starting application threads...")
    
    # Start frame capture thread
    capture_thread = threading.Thread(target=frame_capture_thread)
    capture_thread.daemon = True
    capture_thread.start()
    
    # Start processing thread
    process_thread = threading.Thread(target=process_frames_thread)
    process_thread.daemon = True
    process_thread.start()

    # Start system stats update thread
    stats_thread = threading.Thread(target=update_system_stats)
    stats_thread.daemon = True
    stats_thread.start()
    
    logger.info("All threads started successfully")

if __name__ == '__main__':
    logger.info("Starting Real-time Object Detection with 3D AR")
    logger.info(f"Python version: {platform.python_version()}")
    
    # Check CUDA compatibility
    check_cuda_compatibility()
    
    # Initialize the model
    initialize_model()
    
    # Verify GPU usage
    verify_gpu_usage()
    
    # Setup AR assets
    setup_ar_assets()
    
    # Start background threads
    start_threads()
    
    # Run Flask app with optimized settings
    logger.info("Starting Flask web server...")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)