# Installation Guide for TensorFlow 2 Version

## Quick Install (Recommended)

### Option 1: Using Conda Environment (Best for Most Users)

```bash
# Create new conda environment from file
conda env create -f environment_tf2.yml

# Activate the environment
conda activate RNNExplore_TF2

# Verify installation
python diagnose_tf2.py
```

This creates a separate environment called `RNNExplore_TF2` with:
- Python 3.9
- TensorFlow 2.16.2
- All required dependencies (NumPy, Pandas, SciPy, Matplotlib, etc.)
- TensorBoard for training visualization

**macOS Apple Silicon (M1/M2/M3):** Automatically installs `tensorflow-macos` and `tensorflow-metal` for optimized performance.

---

### Option 2: Using pip

```bash
pip install -r requirements_tf2.txt
```

This will install:
- TensorFlow 2.x
- NumPy, Pandas, SciPy
- Jupyter Notebook
- Matplotlib, Seaborn
- Other dependencies

---

### Option 3: Manual installation

```bash
# Core requirements
pip install tensorflow>=2.4.0
pip install numpy pandas scipy

# Optional but recommended
pip install matplotlib seaborn notebook tqdm scikit-learn pyarrow tensorboard
```

---

## Platform-Specific Notes

### macOS (Apple Silicon - M1/M2/M3)

### macOS (Apple Silicon - M1/M2/M3)

The `environment_tf2.yml` automatically handles this. TensorFlow for Apple Silicon uses Metal for GPU acceleration:

```bash
# Verify Metal GPU support
python -c "import tensorflow as tf; print('Metal devices:', tf.config.list_physical_devices('GPU'))"
```

### Linux/Windows with NVIDIA GPU

For CUDA-enabled GPUs:
```bash
pip install tensorflow[and-cuda]
```

### Check GPU availability:
```bash
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

---

## Verify Installation

After installation, verify everything works:

### 1. Run Diagnostic Script
```bash
python diagnose_tf2.py
```

Should output:
```
✓ TensorFlow 2.16.2 imported successfully
✓ GPU/Metal support: [List of devices]
✓ Can create tensors
✓ Basic operations work
```

### 2. Run Test Script
```bash
python test_tf2_conversion.py
```

You should see:
```
✅ TensorFlow 2 Version: ALL TESTS PASSED
```

### 3. Quick Training Test
```bash
# Edit main_tf2.py to set n_iterations=100 for quick test
python main_tf2.py
```

---

## Troubleshooting

### "No module named 'tensorflow'"
```bash
# Make sure you've activated the correct environment
conda activate RNNExplore_TF2

# If still failing, reinstall
pip install tensorflow
```

### macOS: Segmentation Fault on Import
```bash
# Use the conda environment which handles dependencies correctly
conda env create -f environment_tf2.yml
conda activate RNNExplore_TF2
```

### "No module named 'helpers'"
```bash
# Navigate to the project root directory
cd /path/to/p1_generalization
```

### Version conflicts
```bash
# Create a fresh environment from scratch
conda remove -n RNNExplore_TF2 --all
conda env create -f environment_tf2.yml
conda activate RNNExplore_TF2
```

### Training too slow / Not using GPU
```bash
# Check GPU detection
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# For Apple Silicon: Make sure tensorflow-metal is installed
pip install tensorflow-metal

# For NVIDIA: Install CUDA support
pip install tensorflow[and-cuda]
```

For more issues, see [TROUBLESHOOTING_TF2.md](TROUBLESHOOTING_TF2.md)

## Quick Start After Installation

1. **Test installation:**
   ```bash
   python test_tf2_conversion.py
   ```

2. **Run training:**
   ```bash
   python main_tf2.py
   ```

3. **Monitor training:**
   ```bash
   tensorboard --logdir=tensorboard/
   ```

---

## Recommended: Using a Virtual Environment

### Option A: conda (if you use Anaconda)
```bash
conda env create -f environment_tf2.yml
conda activate RNNExplore_TF2
```

### Option B: venv (built-in Python)
```bash
python3 -m venv venv_tf2
source venv_tf2/bin/activate  # On Windows: venv_tf2\Scripts\activate
pip install -r requirements_tf2.txt
```

---

## Comparing with TF1 Environment

Your original environment (`environment.yml`) uses:
- Python 3.7.2
- TensorFlow 1.15

The new TF2 environment (`environment_tf2.yml`) uses:
- Python 3.9
- TensorFlow 2.4+

**You can keep both!** They're separate environments.

To switch between them:
```bash
conda activate RNNExplore      # Original TF1 environment
conda activate RNNExplore_TF2  # New TF2 environment
```

---

## System Requirements

- **Python**: 3.7 - 3.11
- **OS**: macOS, Linux, or Windows
- **RAM**: 8GB minimum, 16GB recommended
- **Disk**: ~5GB for TensorFlow and dependencies

For GPU support:
- NVIDIA GPU with CUDA Compute Capability 3.5+
- CUDA Toolkit and cuDNN (installed automatically with recent TF versions)
