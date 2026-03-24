@echo off
setlocal EnableExtensions EnableDelayedExpansion

echo.
echo ========================================
echo  StreamDiffusionTD - Mediapipe/NumPy Fix
echo ========================================
echo.

cd /d "%~dp0"

set "VENV_DIR="
if exist "venv\Scripts\python.exe" set "VENV_DIR=venv"
if not defined VENV_DIR if exist ".venv\Scripts\python.exe" set "VENV_DIR=.venv"

if not defined VENV_DIR (
    echo ERROR: venv folder not found in %CD%
    echo Expected either .\venv or .\.venv
    pause
    exit /b 1
)

call "%VENV_DIR%\Scripts\activate.bat"
if not defined VIRTUAL_ENV (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

echo Activated: %VIRTUAL_ENV%
echo.

echo [0/7] Updating pip tooling...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :error

echo.
echo [1/7] Removing conflicting packages...
python -m pip uninstall -y ^
    opencv-python-headless ^
    opencv-contrib-python ^
    opencv-python ^
    mediapipe
if errorlevel 1 goto :error

echo.
echo [2/7] Locking numpy first...
python -m pip install --no-cache-dir --force-reinstall "numpy==1.26.4"
if errorlevel 1 goto :error

echo.
echo [3/7] Installing MediaPipe without dependency drift...
python -m pip install --no-cache-dir --no-deps "mediapipe==0.10.21"
if errorlevel 1 goto :error

echo Installing MediaPipe dependencies manually...
python -m pip install --no-cache-dir ^
    "absl-py>=2.0.0" ^
    "attrs>=23.1.0" ^
    "flatbuffers>=25.2.10" ^
    "matplotlib>=3.7,<3.11" ^
    "protobuf==4.25.3" ^
    "sounddevice>=0.4.4" ^
    "sentencepiece>=0.2.0"
if errorlevel 1 goto :error

echo.
echo [4/7] Installing OpenCV without pulling contrib/headless variants...
python -m pip install --no-cache-dir --no-deps "opencv-python==4.8.1.78"
if errorlevel 1 goto :error

echo.
echo [5/7] Aligning ONNX packages with StreamDiffusion...
python -m pip install --no-cache-dir ^
    "onnx==1.18.0" ^
    "onnxruntime==1.23.2" ^
    "protobuf==4.25.3"
if errorlevel 1 goto :error

echo.
echo [6/7] Reinstalling insightface only if it is present...
python -m pip show insightface >nul 2>nul
if not errorlevel 1 (
    python -m pip uninstall -y insightface
    if errorlevel 1 goto :error

    python -m pip install --no-cache-dir --no-deps "insightface==0.7.3"
    if errorlevel 1 goto :error

    python -m pip install --no-cache-dir --no-deps ^
        "onnxruntime==1.23.2" ^
        "prettytable>=3.0.0"
    if errorlevel 1 goto :error
)

python -m pip uninstall -y opencv-python-headless opencv-contrib-python >nul 2>nul

echo.
echo [7/7] Re-locking numpy and ensuring local package import...
python -m pip install --no-cache-dir --force-reinstall "numpy==1.26.4"
if errorlevel 1 goto :error

python -c "import streamdiffusion" >nul 2>nul
if errorlevel 1 (
    echo streamdiffusion is not installed yet. Installing editable package...
    python -m pip install --no-deps -e .
    if errorlevel 1 goto :error
)

echo.
echo ========================================
echo Verification
echo ========================================
python -c "import numpy; print('numpy ' + numpy.__version__)"
if errorlevel 1 goto :error

python -c "import cv2; print('opencv ' + cv2.__version__)"
if errorlevel 1 goto :error

python -c "import mediapipe as mp; mp.solutions.drawing_utils; print('mediapipe OK')"
if errorlevel 1 goto :error

python -c "import torch; print('torch ' + torch.__version__ + ' CUDA:' + str(torch.cuda.is_available()))"
if errorlevel 1 goto :error

python -c "import streamdiffusion; from streamdiffusion.config import load_config; print('StreamDiffusion OK')"
if errorlevel 1 goto :error

echo.
echo ========================================
echo FIX COMPLETE
echo ========================================
echo.
pause
exit /b 0

:error
echo.
echo ========================================
echo FIX FAILED
echo ========================================
echo Check the error output above.
echo.
pause
exit /b 1
