# Gesture ML System
# =================
# Award-winning quality hand gesture recognition for game UI

## Overview
This system uses machine learning to recognize your custom gestures with high accuracy.

## Files
- `collect_gestures.py` - Record training samples
- `train_gesture_model.py` - Train your custom model
- `gesture_recognizer.py` - Run the trained model

## Quick Start

### Step 1: Collect Training Data
```bash
cd gesture_ml
python collect_gestures.py
```

**Controls:**
- Press `L` - Record a "swipe left" gesture
- Press `R` - Record a "swipe right" gesture
- Press `S` - Record a "select" gesture (pinch)
- Press `I` - Record an "idle" pose
- Press `Q` - Quit

**Goal:** Collect at least 30 samples per gesture (more = better)

**Tips for good training data:**
- Vary your gesture speed (slow, medium, fast)
- Vary the angle slightly
- Include different hand positions
- Do natural, casual gestures

### Step 2: Train the Model
```bash
python train_gesture_model.py
```

This will:
1. Load your recorded data
2. Train an LSTM neural network
3. Export to TensorFlow Lite for fast inference
4. Print accuracy metrics

Target: 90%+ accuracy

### Step 3: Run the Recognizer
```bash
python gesture_recognizer.py
```

Now your gestures are recognized by ML, not thresholds!

## How It Works

### Data Collection
- Each gesture sample is 2 seconds of hand landmark data
- Landmarks are normalized (position, scale invariant)
- Stored as JSON files

### Model Architecture
- **Input:** Sequence of 30 frames × 63 features (21 landmarks × 3 coords)
- **Hidden:** LSTM (64 units) → Dropout → LSTM (32 units) → Dropout
- **Output:** Softmax over gesture classes

### Inference
- TensorFlow Lite for CPU optimization
- Runs at 30+ FPS
- Temporal smoothing prevents flickering
- Confidence threshold prevents false positives

## Customization

### Adding New Gestures
1. Edit `GESTURES` dict in `collect_gestures.py`
2. Edit `GESTURES` list in `train_gesture_model.py`
3. Collect samples for new gesture
4. Re-train

### Tuning
In `gesture_recognizer.py`:
- `CONFIDENCE_THRESHOLD` - Higher = fewer false positives
- `PREDICTION_SMOOTHING` - Higher = more stable but slower
- `GESTURE_COOLDOWN` - Minimum time between triggers

## Requirements
- Python 3.10+
- TensorFlow 2.x
- MediaPipe
- OpenCV

Install:
```bash
pip install tensorflow mediapipe opencv-python
```
