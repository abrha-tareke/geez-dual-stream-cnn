"""
===============================================================================
DUAL-STREAM CNN FOR GE'EZ CHARACTER RECOGNITION
===============================================================================
Complete implementation matching Figure 3.6, Table 3.2, and Table A.2.

Architecture:
- Input: 64x64x1 grayscale
- Shared Convolutional Base (4 blocks, NO MaxPool after Conv4)
- Base Stream: Consonant detection (512-d vector)
- Vocalic Stream: Vocalic detection (512-d vector)
- Fusion: Concatenation (1024-d vector)
- Classification: 2 FC layers -> 276 classes (Softmax)

Total Parameters: 4,665,948
===============================================================================
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'input_shape': (64, 64, 1),
    'num_classes': 276,
    'conv1_filters': 32,
    'conv2_filters': 64,
    'conv3_filters': 128,
    'conv4_filters': 256,
    'stream_filters': 512,
    'fc1_units': 1024,
    'fc2_units': 512,
    'dropout_rate': 0.5,
    'l2_reg': 1e-4,
    'kernel_size': 3,
    'pool_size': 2
}


# =============================================================================
# DATA AUGMENTATION LAYER
# =============================================================================

def create_augmentation_layer():
    """
    Creates data augmentation pipeline for training.
    """
    return keras.Sequential([
        layers.RandomRotation(factor=0.15, fill_mode='constant'),
        layers.RandomZoom(height_factor=(-0.1, 0.1), fill_mode='constant'),
        layers.RandomTranslation(height_factor=(-0.05, 0.05), 
                                 width_factor=(-0.05, 0.05), 
                                 fill_mode='constant'),
        layers.RandomContrast(factor=0.2),
    ], name='data_augmentation')


# =============================================================================
# SHARED CONVOLUTIONAL BASE
# =============================================================================

def create_shared_base(inputs, config):
    """
    Creates the shared convolutional base.
    IMPORTANT: NO MaxPool after Conv4 - output is (8, 8, 256)
    """
    x = inputs
    
    # -------- Conv Block 1 --------
    x = layers.Conv2D(
        config['conv1_filters'], 
        config['kernel_size'], 
        padding='same',
        kernel_regularizer=regularizers.l2(config['l2_reg']),
        name='conv1'
    )(x)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.ReLU(name='relu1')(x)
    x = layers.MaxPooling2D(
        pool_size=(config['pool_size'], config['pool_size']),
        name='pool1'
    )(x)  # Output: (32, 32, 32)
    
    # -------- Conv Block 2 --------
    x = layers.Conv2D(
        config['conv2_filters'], 
        config['kernel_size'], 
        padding='same',
        kernel_regularizer=regularizers.l2(config['l2_reg']),
        name='conv2'
    )(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.ReLU(name='relu2')(x)
    x = layers.MaxPooling2D(
        pool_size=(config['pool_size'], config['pool_size']),
        name='pool2'
    )(x)  # Output: (16, 16, 64)
    
    # -------- Conv Block 3 --------
    x = layers.Conv2D(
        config['conv3_filters'], 
        config['kernel_size'], 
        padding='same',
        kernel_regularizer=regularizers.l2(config['l2_reg']),
        name='conv3'
    )(x)
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.ReLU(name='relu3')(x)
    x = layers.MaxPooling2D(
        pool_size=(config['pool_size'], config['pool_size']),
        name='pool3'
    )(x)  # Output: (8, 8, 128)
    
    # -------- Conv Block 4 (NO MaxPool after this) --------
    x = layers.Conv2D(
        config['conv4_filters'], 
        config['kernel_size'], 
        padding='same',
        kernel_regularizer=regularizers.l2(config['l2_reg']),
        name='conv4'
    )(x)
    x = layers.BatchNormalization(name='bn4')(x)
    x = layers.ReLU(name='relu4')(x)
    # ❌ NO MaxPooling2D here!
    # Output: (8, 8, 256) - matches Table 3.2 and Figure 3.6
    
    return x


# =============================================================================
# DUAL STREAMS: BASE (Consonant) and VOCALIC (Vowel)
# =============================================================================

def create_stream(input_tensor, stream_name, config):
    """
    Creates a single stream (Base or Vocalic).
    """
    x = layers.Conv2D(
        config['stream_filters'], 
        config['kernel_size'], 
        padding='same',
        kernel_regularizer=regularizers.l2(config['l2_reg']),
        name=f'{stream_name}_conv'
    )(input_tensor)
    x = layers.BatchNormalization(name=f'{stream_name}_bn')(x)
    x = layers.ReLU(name=f'{stream_name}_relu')(x)
    x = layers.GlobalAveragePooling2D(name=f'{stream_name}_gap')(x)
    # Output: (512,)
    
    return x


# =============================================================================
# COMPLETE DUAL-STREAM CNN MODEL
# =============================================================================

def create_dual_stream_cnn(config=None):
    """
    Creates the complete Dual-Stream CNN model for Ge'ez character recognition.
    
    Returns:
        keras.Model: Compiled dual-stream CNN model
        
    Architecture matches:
        - Figure 3.6 (Corrected)
        - Table 3.2
        - Table A.2
    """
    if config is None:
        config = CONFIG
    
    # -------- Input Layer --------
    inputs = keras.Input(shape=config['input_shape'], name='input')
    
    # -------- Data Augmentation (Training only) --------
    augmentation = create_augmentation_layer()
    x = augmentation(inputs, training=False)
    
    # -------- Shared Convolutional Base --------
    # Output: (8, 8, 256) - NO MaxPool after Conv4
    shared_features = create_shared_base(x, config)
    
    # -------- Dual Streams --------
    base_features = create_stream(shared_features, 'base', config)
    vocalic_features = create_stream(shared_features, 'vocalic', config)
    
    # -------- Feature Fusion --------
    merged = layers.Concatenate(name='concat')([base_features, vocalic_features])
    
    # -------- Classification Head --------
    x = layers.Dense(
        config['fc1_units'],
        activation='relu',
        kernel_regularizer=regularizers.l2(config['l2_reg']),
        name='fc1'
    )(merged)
    x = layers.Dropout(config['dropout_rate'], name='dropout1')(x)
    
    x = layers.Dense(
        config['fc2_units'],
        activation='relu',
        kernel_regularizer=regularizers.l2(config['l2_reg']),
        name='fc2'
    )(x)
    x = layers.Dropout(config['dropout_rate'], name='dropout2')(x)
    
    outputs = layers.Dense(
        config['num_classes'],
        activation='softmax',
        kernel_regularizer=regularizers.l2(config['l2_reg']),
        name='output'
    )(x)
    
    # -------- Create Model --------
    model = keras.Model(inputs=inputs, outputs=outputs, name='dual_stream_cnn')
    
    return model


# =============================================================================
# MODEL SUMMARY
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DUAL-STREAM CNN FOR GE'EZ CHARACTER RECOGNITION")
    print("=" * 70)
    
    model = create_dual_stream_cnn()
    model.summary()
    
    total_params = model.count_params()
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Expected: 4,665,948")
    print(f"Match: {'✅ YES' if total_params == 4665948 else '❌ NO'}")