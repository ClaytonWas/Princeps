# Princeps Gesture Recognition - Setup Script
# Run this script to set up the project

Write-Host "================================" -ForegroundColor Cyan
Write-Host "  Princeps Setup" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Check Python
Write-Host "Checking Python..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python not found. Please install Python 3.10+ first." -ForegroundColor Red
    exit 1
}
Write-Host "Found: $pythonVersion" -ForegroundColor Green

# Create virtual environment
Write-Host ""
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path ".venv") {
    Write-Host "Virtual environment already exists, skipping..." -ForegroundColor Gray
} else {
    python -m venv .venv
    Write-Host "Created .venv" -ForegroundColor Green
}

# Activate virtual environment
Write-Host ""
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host ""
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install dependencies
Write-Host ""
Write-Host "Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

# Download hand landmarker model if needed
Write-Host ""
Write-Host "Checking for hand landmarker model..." -ForegroundColor Yellow
$modelPath = "gesture_ml\hand_landmarker.task"
if (Test-Path $modelPath) {
    Write-Host "Model already exists" -ForegroundColor Green
} else {
    Write-Host "Downloading hand landmarker model..." -ForegroundColor Yellow
    $url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    Invoke-WebRequest -Uri $url -OutFile $modelPath
    Write-Host "Downloaded!" -ForegroundColor Green
}

Write-Host ""
Write-Host "================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To activate the environment:" -ForegroundColor White
Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor Gray
Write-Host ""
Write-Host "To collect gesture training data:" -ForegroundColor White
Write-Host "  python gesture_ml\collect_gestures.py" -ForegroundColor Gray
Write-Host ""
Write-Host "To train the model:" -ForegroundColor White
Write-Host "  python gesture_ml\train_gesture_model.py" -ForegroundColor Gray
Write-Host ""
Write-Host "To run gesture recognition:" -ForegroundColor White
Write-Host "  python gesture_ml\gesture_recognizer.py" -ForegroundColor Gray
Write-Host ""
