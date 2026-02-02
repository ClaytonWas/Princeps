# Princeps

Hand gesture recognition for game UI.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## ./gesture_ml

ML Training Library For Gestures

1. **Collect gestures** (30+ samples each)
   ```
   python gesture_ml/collect_gestures.py
   ```
   Keys: L=left, R=right, S=select, I=idle, Q=quit

2. **Train model**
   ```
   python gesture_ml/train_gesture_model.py
   ```

3. **Run**
   ```
   python gesture_ml/gesture_recognizer.py
   ```
