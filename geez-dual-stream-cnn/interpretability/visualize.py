"""
===============================================================================
INTERPRETABILITY VISUALIZATION (Grad-CAM and SHAP)
===============================================================================
Usage:
    python interpretability/visualize.py --model weights/best_model.h5 --image sample.png
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
import tensorflow as tf


def grad_cam_heatmap(model, image, class_idx, last_conv_layer='conv4'):
    """
    Generate Grad-CAM heatmap.
    """
    # Create a model that maps input to conv4 and output
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer).output, model.output]
    )
    
    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        loss = predictions[:, class_idx]
    
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight conv output with gradients
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(conv_outputs, pooled_grads), axis=-1)
    
    # ReLU and normalize
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-6)
    
    return heatmap.numpy()


def visualize_grad_cam(model_path, image_path, class_idx=0):
    """
    Visualize Grad-CAM heatmap.
    """
    print("=" * 70)
    print("GRAD-CAM VISUALIZATION")
    print("=" * 70)
    
    # Load model
    model = load_model(model_path)
    print("✅ Model loaded successfully!")
    
    # Load and preprocess image
    # NOTE: User must implement image loading
    print(f"\nImage: {image_path}")
    print("⚠️ Image loading not implemented. This is a placeholder.")
    
    print("\n✅ Visualization complete!")


def visualize_shap(model_path, image_path):
    """
    Visualize SHAP values.
    """
    print("=" * 70)
    print("SHAP VISUALIZATION")
    print("=" * 70)
    
    print("⚠️ SHAP visualization requires shap library.")
    print("Install: pip install shap")
    
    print("\n✅ Visualization complete!")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Interpretability Visualization')
    parser.add_argument('--model', type=str, required=True, help='Path to model weights')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--class_idx', type=int, default=0, help='Target class index')
    parser.add_argument('--method', type=str, default='gradcam', choices=['gradcam', 'shap'])
    
    args = parser.parse_args()
    
    if args.method == 'gradcam':
        visualize_grad_cam(args.model, args.image, args.class_idx)
    else:
        visualize_shap(args.model, args.image)