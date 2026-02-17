"""
Simple diagnostic script to identify TensorFlow installation issues.
"""

import sys
print("Python version:", sys.version)
print("=" * 80)

# Test 1: Basic imports
print("Test 1: Importing basic modules...")
try:
    import numpy as np
    print("✓ NumPy imported successfully. Version:", np.__version__)
except Exception as e:
    print(f"✗ NumPy import failed: {e}")
    sys.exit(1)

try:
    import pandas as pd
    print("✓ Pandas imported successfully. Version:", pd.__version__)
except Exception as e:
    print(f"✗ Pandas import failed: {e}")

try:
    import scipy
    print("✓ SciPy imported successfully. Version:", scipy.__version__)
except Exception as e:
    print(f"✗ SciPy import failed: {e}")

print("\n" + "=" * 80)
print("Test 2: Importing TensorFlow...")
print("=" * 80)

try:
    import tensorflow as tf
    print("✓ TensorFlow imported successfully!")
    print(f"  Version: {tf.__version__}")
    print(f"  Keras version: {tf.keras.__version__}")
except ImportError as e:
    print(f"✗ TensorFlow not installed: {e}")
    print("\nInstall with: pip install tensorflow")
    sys.exit(1)
except Exception as e:
    print(f"✗ TensorFlow import failed with error: {e}")
    print("\nThis might be a compatibility issue.")
    sys.exit(1)

print("\n" + "=" * 80)
print("Test 3: TensorFlow basic operations...")
print("=" * 80)

try:
    # Simple tensor operation
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
    c = tf.matmul(a, b)
    print("✓ Matrix multiplication successful")
    print(f"  Result shape: {c.shape}")
except Exception as e:
    print(f"✗ TensorFlow operations failed: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("Test 4: GPU/Device detection...")
print("=" * 80)

try:
    gpus = tf.config.list_physical_devices('GPU')
    print(f"  GPUs available: {len(gpus)}")
    if gpus:
        for gpu in gpus:
            print(f"    - {gpu}")
    else:
        print("    (No GPUs detected - using CPU)")
    
    cpus = tf.config.list_physical_devices('CPU')
    print(f"  CPUs available: {len(cpus)}")
except Exception as e:
    print(f"⚠ Device detection had issues: {e}")

print("\n" + "=" * 80)
print("Test 5: Keras layers...")
print("=" * 80)

try:
    from tensorflow.keras import layers
    dense = layers.Dense(10, activation='relu')
    print("✓ Keras layers imported successfully")
except Exception as e:
    print(f"✗ Keras layers import failed: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ ALL BASIC TESTS PASSED")
print("=" * 80)
print("\nTensorFlow is working correctly!")
print("You can now proceed to test the full conversion.")
print("\nNext step: Check if project-specific imports work")
