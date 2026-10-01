@echo off
REM Install ultralytics (YOLO) into the existing PyTorch venv at D:\mnist_torch
setlocal
cd /d "%~dp0"

set "VENV="
if exist ".venv\Scripts\activate.bat" set "VENV=.venv"
if exist "venv\Scripts\activate.bat" set "VENV=venv"
if exist "env\Scripts\activate.bat" set "VENV=env"

if "%VENV%"=="" (
  echo [ERROR] No virtualenv found under D:\mnist_torch
  echo Expected one of: .venv, venv, or env
  echo Create one first, e.g.:
  echo   python -m venv .venv
  echo   .venv\Scripts\activate
  echo   pip install torch torchvision
  exit /b 1
)

echo [INFO] Using virtualenv: %VENV%
call "%VENV%\Scripts\activate.bat"
if errorlevel 1 (
  echo [ERROR] Failed to activate %VENV%
  exit /b 1
)

echo [INFO] Python:
python -c "import sys; print(sys.executable)"
echo [INFO] Installing ultralytics from requirements.txt ...
python -m pip install --upgrade pip
python -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 (
  echo [ERROR] pip install failed
  exit /b 1
)

echo.
echo [OK] ultralytics installed.
echo Run classification test:
echo   python test_classify.py
echo Or with a local image:
echo   python test_classify.py path\to\image.jpg
endlocal
