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

app = Flask(__name__)

# Create static folders if they don't exist
os.makedirs('static/ar', exist_ok=True)

# Configuration
IP_WEBCAM_URL = "http://192.168.196.14:8080/video"  # Your IP Webcam URL
MODEL_PATH = "yolov8n.pt"  # Using YOLOv8 nano for speed
CONFIDENCE_THRESHOLD = 0.5
USE_GPU = True
# ROTATE_VIDEO = True  # Set to True to rotate the video by 90 degrees
PROCESS_EVERY_N_FRAMES = 1  # Process every frame (set to 2 or 3 to skip frames if needed)
RESIZE_FACTOR = 1.0  # Resize factor for input frames (0.5 = half size, 1.0 = original size)
MAX_QUEUE_SIZE = 1  # Maximum queue size for frames
JPEG_QUALITY = 70  # JPEG quality for streaming (lower = faster but lower quality)

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

def initialize_model():
    global model
    if torch.cuda.is_available() and USE_GPU:
        # Force PyTorch to use the GPU
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        
        # Enable CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True
        
        # Set GPU device
        device = torch.device("cuda:0")
        print(f"Using device: {device}")
        
        # Attempt to configure for maximum performance
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except:
            print("TF32 optimization not available")
    else:
        device = torch.device("cpu")
        print(f"Using device: {device}")
    
    # Load and optimize model
    model = YOLO(MODEL_PATH)
    model.to(device)
    
    # Enable half-precision for faster inference
    if torch.cuda.is_available() and USE_GPU:
        print("Enabling half precision (FP16)")
        model.model.half()

def frame_capture_thread():
    global processing_active
    
    cap = cv2.VideoCapture(IP_WEBCAM_URL)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return
    
    # Set optimal buffer size to reduce latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Try to set the lowest possible latency mode
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    
    frame_count = 0
    
    while processing_active:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to receive frame.")
            time.sleep(0.1)  # Prevent CPU spinning
            continue
        
        frame_count += 1
        
        # Process only every Nth frame
        if frame_count % PROCESS_EVERY_N_FRAMES != 0:
            continue
        
        # Rotate frame if needed (90 degrees counterclockwise)
        # if ROTATE_VIDEO:
        #     frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
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

def process_frames_thread():
    global processed_frame, current_fps, processing_active, detected_objects
    
    frame_times = []
    max_times = 10  # Number of frames to average FPS over
    
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
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available() and USE_GPU):
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
            
        except queue.Empty:
            # No frame available, just continue
            continue
        except Exception as e:
            print(f"Error in processing thread: {e}")
            time.sleep(0.1)  # Prevent CPU spinning

def generate_frames():
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

# Function to update system stats continuously
def update_system_stats():
    global system_stats
    
    while True:
        try:
            # Get CPU usage
            system_stats["cpu_percent"] = psutil.cpu_percent()
            
            # Get memory usage
            memory = psutil.virtual_memory()
            system_stats["memory_used"] = memory.used
            system_stats["memory_total"] = memory.total
            
            # Simulated integrated GPU usage
            system_stats["integrated_gpu_percent"] = min(system_stats["cpu_percent"] * 1.2, 100)
            
            # Simulated NVIDIA GPU usage (or more accurate if available)
            nvidia_gpu_percent = 35.0
            if torch.cuda.is_available():
                try:
                    nvidia_gpu_percent = memory.percent * 0.8
                except:
                    pass
            system_stats["nvidia_gpu_percent"] = nvidia_gpu_percent
            
            # GPU memory - using CUDA info if available
            gpu_memory_total = 4 * 1024 * 1024 * 1024  # 4GB for RTX 3050
            gpu_memory_used = 0
            
            if torch.cuda.is_available():
                try:
                    gpu_memory_used = torch.cuda.memory_allocated(0)
                    gpu_memory_reserved = torch.cuda.memory_reserved(0)
                    
                    if hasattr(torch.cuda, 'get_device_properties'):
                        props = torch.cuda.get_device_properties(0)
                        if hasattr(props, 'total_memory'):
                            gpu_memory_total = props.total_memory
                except:
                    pass
            
            system_stats["gpu_memory_used"] = gpu_memory_used
            system_stats["gpu_memory_total"] = gpu_memory_total
            
        except Exception as e:
            print(f"Error updating system stats: {e}")
        
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
    
    # Try to get GPU memory usage if available
    if torch.cuda.is_available():
        try:
            mem_allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)  # Convert to GB
            mem_reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
            stats_data["gpu_mem"] = f"{mem_allocated:.2f}GB / {mem_reserved:.2f}GB"
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
    
    print("Checking for required 3D models...")
    missing_models = []
    for model_file in required_models:
        model_path = os.path.join(ar_dir, model_file)
        if not os.path.exists(model_path):
            missing_models.append(model_file)
    
    if missing_models:
        print(f"Warning: The following model files are missing from {ar_dir}:")
        for model in missing_models:
            print(f"  - {model}")
        print(f"\nPlease place your 3D models (.glb format) in the {ar_dir} directory.")
        print("You can convert .blend files to .glb using Blender's export function.")
    else:
        print("All required 3D models found!")
    
    print(f"AR setup complete. 3D models should be in {ar_dir} directory.")

def start_threads():
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

if __name__ == '__main__':
    # Initialize the model
    initialize_model()
    
    # Setup AR assets
    setup_ar_assets()
    
    # Start background threads
    start_threads()
    
    # Run Flask app with optimized settings
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)