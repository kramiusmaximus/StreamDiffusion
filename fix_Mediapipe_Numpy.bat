@echo offecho.echo ========================================echo  StreamDiffusionTD - Mediapipe/NumPy Fixecho ========================================echo.
cd /d "%~dp0"if not exist "venv" (    echo ERROR: venv folder not found in %CD%    pause    exit /b 1)
call "venv\Scripts\activate.bat"if "%VIRTUAL_ENV%" == "" (    echo ERROR: Failed to activate venv    pause    exit /b 1)echo Activated: %VIRTUAL_ENV%
echo.echo [1/6] Removing conflicting packages...pip uninstall -y opencv-python-headless mediapipe opencv-python opencv-contrib-python
echo.echo [2/6] Locking numpy FIRST (prevents drift)...pip install --no-cache-dir "numpy==1.26.4" --force-reinstall
echo.echo [3/6] Installing mediapipe with --no-deps (prevents numpy upgrade)...pip install --no-cache-dir --no-deps "mediapipe==0.10.18"echo Installing mediapipe dependencies manually...pip install --no-cache-dir absl-py attrs flatbuffers matplotlib protobuf sounddevice
echo.echo [4/6] Installing opencv-python with --no-deps...pip install --no-cache-dir --no-deps "opencv-python==4.8.1.78"
echo.echo [5/6] Fixing insightface (if present)...pip show insightface >nul 2>&1 && (    pip uninstall -y insightface    pip install --no-cache-dir --no-deps insightface    pip install --no-cache-dir --no-deps onnxruntime prettytable)pip uninstall -y opencv-python-headless opencv-contrib-python 2>nul
echo.echo [6/6] Fixing onnx version (onnx_graphsurgeon compatibility)...pip install --no-cache-dir "onnx<1.20.0"
echo.echo Re-locking numpy (final safety check)...pip install --no-cache-dir "numpy==1.26.4" --force-reinstall
echo.echo ========================================echo Verificationecho ========================================python -c "import numpy; print(f'numpy {numpy.__version__}')"python -c "import cv2; print(f'opencv {cv2.__version__}')"python -c "import mediapipe as mp; mp.solutions.drawing_utils; print('mediapipe OK')" || goto :errorpython -c "import torch; print(f'torch {torch.__version__} CUDA:{torch.cuda.is_available()}')"python -c "from streamdiffusion.config import load_config; print('StreamDiffusion OK')" || echo StreamDiffusion FAILED

