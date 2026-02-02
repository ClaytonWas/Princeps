"""
Gesture Model Training
======================
Trains a custom gesture classifier using the collected data.

Uses a simple but effective approach:
- LSTM network for temporal sequence classification
- Works with the landmark data from MediaPipe
- Exports to TensorFlow Lite for fast inference

Prerequisites:
- Run collect_gestures.py first to gather training data
- Need at least 30 samples per gesture class

Output:
- gesture_model.tflite - The trained model
- gesture_labels.json - Label mapping
"""

import os
import json
import numpy as np
from pathlib import Path

# Check for TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    print(f"TensorFlow version: {tf.__version__}")
except ImportError:
    print("TensorFlow not installed. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "tensorflow"])
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

# =====================
# CONFIG
# =====================
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "gesture_data"
MODEL_OUTPUT = SCRIPT_DIR / "gesture_model"
SEQUENCE_LENGTH = 30  # Fixed sequence length for LSTM
NUM_LANDMARKS = 21
NUM_COORDS = 3  # x, y, z

GESTURES = ['swipe_left', 'swipe_right', 'select', 'idle']


def load_samples(gesture_name):
    """Load all samples for a gesture."""
    path = DATA_DIR / gesture_name
    samples = []
    
    for file in path.glob("*.json"):
        with open(file) as f:
            data = json.load(f)
        
        frames = data.get("frames", [])
        if len(frames) < 5:
            continue
        
        # Extract landmark sequences
        sequence = []
        for frame in frames:
            landmarks = frame.get("landmarks")
            if landmarks:
                sequence.append(landmarks)
        
        if len(sequence) >= 5:
            samples.append(sequence)
    
    return samples


def pad_or_truncate(sequence, target_length):
    """Pad or truncate sequence to fixed length."""
    seq = np.array(sequence)
    
    if len(seq) >= target_length:
        # Take evenly spaced frames
        indices = np.linspace(0, len(seq) - 1, target_length, dtype=int)
        return seq[indices]
    else:
        # Pad with last frame
        padding = np.tile(seq[-1], (target_length - len(seq), 1))
        return np.vstack([seq, padding])


def prepare_dataset():
    """Load and prepare the full dataset."""
    X = []
    y = []
    
    for gesture_idx, gesture_name in enumerate(GESTURES):
        samples = load_samples(gesture_name)
        print(f"  {gesture_name}: {len(samples)} samples")
        
        for sample in samples:
            processed = pad_or_truncate(sample, SEQUENCE_LENGTH)
            X.append(processed)
            y.append(gesture_idx)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return X, y


def create_model(input_shape, num_classes):
    """Create the LSTM gesture classifier."""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # LSTM layers
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(32),
        layers.Dropout(0.3),
        
        # Dense layers
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train():
    """Main training function."""
    print("\n" + "="*50)
    print("GESTURE MODEL TRAINING")
    print("="*50)
    
    # Load data
    print("\nLoading data...")
    X, y = prepare_dataset()
    print(f"\nTotal samples: {len(X)}")
    print(f"Input shape: {X.shape}")
    
    if len(X) < 40:
        print("\n⚠ WARNING: Not enough training data!")
        print("Collect at least 30 samples per gesture.")
        return
    
    # Split train/val
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Create model
    print("\nCreating model...")
    input_shape = (SEQUENCE_LENGTH, NUM_LANDMARKS * NUM_COORDS)
    model = create_model(input_shape, len(GESTURES))
    model.summary()
    
    # Train
    print("\nTraining...")
    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            factor=0.5,
            patience=5
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=16,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\nEvaluating...")
    loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Accuracy: {accuracy:.1%}")
    
    # Save Keras model
    model_path = Path(MODEL_OUTPUT)
    model_path.mkdir(exist_ok=True)
    model.save(model_path / "gesture_model.keras")
    print(f"\n✓ Saved Keras model to {model_path / 'gesture_model.keras'}")
    
    # Convert to TFLite
    print("\nConverting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # Required for LSTM models in TFLite
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter._experimental_lower_tensor_list_ops = False
    tflite_model = converter.convert()
    
    tflite_path = model_path / "gesture_model.tflite"
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"✓ Saved TFLite model to {tflite_path}")
    
    # Save labels
    labels_path = model_path / "gesture_labels.json"
    with open(labels_path, 'w') as f:
        json.dump({i: name for i, name in enumerate(GESTURES)}, f, indent=2)
    print(f"✓ Saved labels to {labels_path}")
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"\nModel accuracy: {accuracy:.1%}")
    print(f"\nFiles created:")
    print(f"  - {model_path / 'gesture_model.keras'}")
    print(f"  - {tflite_path}")
    print(f"  - {labels_path}")
    print("\nNext step: Run gesture_recognizer.py to use the model")


if __name__ == "__main__":
    train()
