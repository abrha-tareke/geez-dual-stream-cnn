"""
===============================================================================
SAMPLE DATA GENERATOR FOR TESTING
===============================================================================
This script generates synthetic Ge'ez-like character images for testing
the model architecture and pipeline. It does NOT contain real dataset images.
===============================================================================
"""

import numpy as np
import cv2
import os


def generate_synthetic_geez_image(shape=(64, 64)):
    """
    Generate a synthetic character-like image for testing purposes only.
    This is NOT a real Ge'ez character.
    """
    img = np.ones(shape, dtype=np.uint8) * 255
    
    # Draw a random shape mimicking a character
    center = (32, 32)
    radius = np.random.randint(10, 20)
    
    # Draw a circle or ellipse
    axes = (radius, radius // 2)
    angle = np.random.randint(0, 180)
    cv2.ellipse(img, center, axes, angle, 0, 360, 0, -1)
    
    # Add some noise
    noise = np.random.normal(0, 10, shape).astype(np.uint8)
    img = cv2.add(img, noise)
    
    return img


def generate_sample_dataset(num_samples=100, num_classes=10):
    """
    Generate a small sample dataset for testing.
    """
    images = []
    labels = []
    
    for i in range(num_samples):
        img = generate_synthetic_geez_image()
        images.append(img)
        labels.append(np.random.randint(0, num_classes))
    
    return np.array(images), np.array(labels)


if __name__ == "__main__":
    print("=" * 70)
    print("SAMPLE DATA GENERATOR")
    print("=" * 70)
    
    print("\nGenerating sample test data...")
    X, y = generate_sample_dataset(100, 10)
    
    print(f"Sample shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Class distribution: {np.unique(y, return_counts=True)}")
    
    print("\n✅ Sample data generated successfully!")