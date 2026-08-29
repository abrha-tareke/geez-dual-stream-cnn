"""
===============================================================================
EVALUATION SCRIPT FOR DUAL-STREAM CNN
===============================================================================
Usage:
    python evaluation/evaluate.py --model weights/best_model.h5
===============================================================================
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensorflow.keras.models import load_model


def evaluate_model(model_path, test_data=None):
    """
    Evaluate a trained model.
    """
    print("=" * 70)
    print("EVALUATING DUAL-STREAM CNN")
    print("=" * 70)
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    model = load_model(model_path)
    print("✅ Model loaded successfully!")
    
    # Model summary
    model.summary()
    
    # If test data provided, evaluate
    if test_data is not None:
        X_test, y_test = test_data
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest Accuracy: {acc*100:.2f}%")
        print(f"Test Loss: {loss:.4f}")
    
    print("\n✅ Evaluation complete!")
    return model


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Dual-Stream CNN')
    parser.add_argument('--model', type=str, required=True, help='Path to model weights')
    
    args = parser.parse_args()
    
    evaluate_model(args.model)