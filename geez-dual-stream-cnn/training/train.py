"""
===============================================================================
TRAINING SCRIPT FOR DUAL-STREAM CNN
===============================================================================
Usage:
    python training/train.py --sample_data
    python training/train.py --data_path /path/to/data
===============================================================================
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dual_stream_cnn import create_dual_stream_cnn, CONFIG


# =============================================================================
# DATA LOADER
# =============================================================================

def load_sample_data():
    """
    Load sample data for testing.
    """
    try:
        from data.sample_data import generate_sample_dataset
        X, y = generate_sample_dataset(num_samples=500, num_classes=276)
        
        # Simple train/val split
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"Sample data: {len(X_train)} train, {len(X_val)} val")
        
        return (X_train, y_train), (X_val, y_val)
    
    except ImportError:
        print("❌ sample_data.py not found. Please create it in the data/ folder.")
        return None, None


def load_real_data(data_path):
    """
    Load real Ge'ez dataset.
    NOTE: This is a placeholder. Users must implement their own data loading.
    """
    print(f"Loading data from: {data_path}")
    print("⚠️ Please implement your own data loading logic.")
    
    # Example structure (user must replace):
    # X_train = np.load(os.path.join(data_path, 'X_train.npy'))
    # y_train = np.load(os.path.join(data_path, 'y_train.npy'))
    # ...
    
    return None, None


# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_model(data_path=None, sample_data=False, epochs=10, batch_size=32):
    """
    Train the dual-stream CNN model.
    """
    print("=" * 70)
    print("TRAINING DUAL-STREAM CNN FOR GE'EZ CHARACTER RECOGNITION")
    print("=" * 70)
    
    # Load data
    if sample_data:
        print("\n📊 Using sample data...")
        (X_train, y_train), (X_val, y_val) = load_sample_data()
    elif data_path:
        print(f"\n📊 Loading data from: {data_path}")
        (X_train, y_train), (X_val, y_val) = load_real_data(data_path)
    else:
        print("\n⚠️ No data path provided. Using sample data...")
        (X_train, y_train), (X_val, y_val) = load_sample_data()
    
    if X_train is None:
        print("❌ Failed to load data. Exiting.")
        return
    
    # One-hot encode labels
    y_train = keras.utils.to_categorical(y_train, num_classes=276)
    y_val = keras.utils.to_categorical(y_val, num_classes=276)
    
    # Normalize images
    X_train = X_train.astype('float32') / 255.0
    X_val = X_val.astype('float32') / 255.0
    
    # Add channel dimension if needed
    if len(X_train.shape) == 3:
        X_train = X_train[..., np.newaxis]
        X_val = X_val[..., np.newaxis]
    
    print(f"\nTraining data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    
    # Create model
    model = create_dual_stream_cnn()
    print("\n✅ Model created successfully!")
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    
    test_loss, test_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Accuracy: {test_acc*100:.2f}%")
    
    print("\n✅ Training completed successfully!")
    return model, history


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Dual-Stream CNN')
    parser.add_argument('--data_path', type=str, help='Path to dataset')
    parser.add_argument('--sample_data', action='store_true', help='Use sample data')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    train_model(
        data_path=args.data_path,
        sample_data=args.sample_data,
        epochs=args.epochs,
        batch_size=args.batch_size
    )