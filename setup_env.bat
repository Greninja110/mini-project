@echo off
echo Setting up environment for YOLO object detection with GPU support
echo ===============================================================

REM Create logs directory
if not exist logs mkdir logs

REM Check for NVIDIA drivers
echo Checking NVIDIA drivers...
nvidia-smi 
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: NVIDIA drivers not found or not working properly!
    echo Please download and install the latest drivers from:
    echo https://www.nvidia.com/download/index.aspx
    pause
    exit /b
)

REM Check PyTorch CUDA
echo Checking PyTorch CUDA support...
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('GPU device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: PyTorch check failed. Make sure PyTorch is installed correctly.
    pause
    exit /b
)

echo.
echo Setup complete. Ready to run the application.
echo.
echo Next steps:
echo 1. Run 'python app.py' to start the application
echo 2. Check logs in the logs directory for debugging information
echo.
pause