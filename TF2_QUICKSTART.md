# TensorFlow 2 Quick Start Guide

## Overview

This guide helps you quickly get started with the TensorFlow 2 implementation of the Actor-Critic RNN for bandit tasks.

## Installation

### Option 1: Conda Environment (Recommended)

Create and activate the environment from the provided YAML file:

```bash
# Create environment from file
conda env create -f environment_tf2.yml

# Activate the environment
conda activate RNNExplore_TF2
```

The environment includes:
- Python 3.9
- TensorFlow 2.16.2 (tensorflow-macos for Apple Silicon, or standard tensorflow)
- NumPy, Pandas, SciPy
- Matplotlib for plotting
- TensorBoard for monitoring

**macOS Apple Silicon Note:** The environment automatically installs `tensorflow-macos` and `tensorflow-metal` for optimized performance on M1/M2/M3 chips.

### Option 2: pip Installation

Alternatively, install using pip:

```bash
pip install -r requirements_tf2.txt
```

### Verify Installation

```bash
# Check TensorFlow version
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"

# Check for GPU/Metal support (macOS)
python -c "import tensorflow as tf; print('Metal devices:', tf.config.list_physical_devices('GPU'))"

# Run diagnostic script
python diagnose_tf2.py
```

## Quick Test

Run the test script to verify everything is working:
```bash
python test_tf2_conversion.py
```

This will:
- ✓ Check TensorFlow 2 installation
- ✓ Test network creation
- ✓ Verify forward pass
- ✓ Validate outputs

## Running Training

### Basic Usage

```bash
python main_tf2.py
```

### Configuration

Edit `main_tf2.py` to customize:

```python
# Number of hidden units to test
N_HIDDEN = [48, 64, 80]

# Entropy regularization coefficients
ENTROPIES = [0, 0.05, 'linear']  # 'linear' = anneal from 1 to 0

# Model instance IDs (for parallel runs)
IDS = [0, 1, 2, 3]

# Training flag
TRAIN_RNN = True

# Testing configuration
TEST_DAW = False  # True = test on Daw walks, False = test on fixed bandits
TEST_SD = [0.1]  # Standard deviations to test
```

## Training Configuration Details

### RNN Types
```python
rnn_type='lstm'   # Standard LSTM
rnn_type='lstm2'  # Custom LSTM with noise injection
rnn_type='rnn'    # Elman RNN
```

### Noise Types
```python
noise='none'              # No computational noise
noise='update-dependant'  # Weber noise (Findling & Wyart, 2020)
noise='constant'          # Not yet implemented
```

### Learning Algorithms
```python
learning_algorithm='a2c'       # Advantage Actor-Critic
learning_algorithm='reinforce' # REINFORCE
```

## Example: Training a Single Model

```python
from classes.bandits import bandit_class as bc
from classes.neural_networks import p_network_class_organized_lstm_cell_tf2 as nn

# Create bandit
bandit = bc.Bandit(
    bandit_type='restless',
    arms=4,
    num_steps=300,
    reward_type='continuous',
    noise_sd=0.1,
    punish=True
)

# Create and train network
nnet = nn.neural_network(
    bandit=bandit,
    noise='update-dependant',
    noise_parameter=0.5,
    discount_rate=0.5,
    value_loss_weight=0.5,
    entropy_loss_weight=0.05,
    rnn_type='lstm2',
    learning_algorithm='a2c',
    n_iterations=50000,
    n_hidden_neurons=64,
    model_id=0
)

# Train
nnet.train()

# Test
nnet.test(
    bandit='fixed_res_rt_con_p_{}_a_4_n_300_run_{}.zip',
    bandit_param_range=[0.1],
    n_runs=1
)
```

## Monitoring Training

### TensorBoard

Start TensorBoard to monitor training:
```bash
tensorboard --logdir=tensorboard/
```

Then open your browser to: `http://localhost:6006`

### Metrics Available
- **Perf/Reward**: Mean episode reward
- **Losses/Policy Loss**: Policy gradient loss
- **Losses/Entropy**: Entropy regularization
- **Losses/Var Norm**: Parameter norm
- **Gradnorm**: Gradient norm (for detecting instabilities)

## Saved Models

Models are saved in `saved_models/` with naming convention:
```
{rnn_type}_{algorithm}_nh_{hidden}_lr_{lr}_n_{noise}_p_{noise_param}_...
```

Example:
```
lstm2_a2c_nh_64_lr_0001_n_u_p_05_ew_005_vw_05_dr_05_res_d_01_rt_con_a_4_n_300_te_50000_id_0
```

## Testing Models

### Test on Daw Walks
```python
nnet.test(
    bandit='classes/bandits/Daw2006_payoffs1.csv',
    bandit_param_range=[1],
    n_runs=1
)
```

### Test on Fixed Bandits
```python
nnet.test(
    bandit='fixed_res_rt_con_p_{}_a_4_n_300_run_{}.zip',
    bandit_param_range=[0.05, 0.1, 0.15],
    n_runs=5
)
```

## Output Files

### Training Outputs
- **Checkpoints**: `saved_models/{model_name}/`
- **TensorBoard Logs**: `tensorboard/{model_name}/`

### Test Outputs
- **Test Results**: `data/rnn_raw_data/{model_name}_test_b_{bandit}_p_{param}`
- **Format**: Pickled pandas MultiIndex DataFrame

### Test DataFrame Structure

**Indices**:
- `rnn_type`: RNN architecture
- `learning_algorithm`: Learning algorithm
- `noise`: Noise type
- `train_sd`: Training noise SD
- `rnn_id`: Model ID
- `test_sd`: Testing noise SD
- `run`: Run number
- `reward_instance`: Reward instance

**Columns**:
- `choice`: Actions taken
- `reward`: Rewards received
- `value`: Estimated state values
- `p_rew_1`, `p_rew_2`, ...: Reward probabilities
- `softmax_1`, `softmax_2`, ...: Action probabilities
- `rnn_state_1`, `rnn_state_2`, ...: Hidden states
- `added_noise_rnn_state_1`, ...: Noise added to hidden states
- `accuracy`: Binary indicator of optimal choice
- `is_switch`: Binary indicator of action switch

## Performance Tips

### 1. Enable GPU Memory Growth
```python
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

### 2. Use Mixed Precision (Modern GPUs)
```python
from tensorflow.keras import mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
```

### 3. Profile Your Code
```python
# Add to training code
tf.profiler.experimental.start('logdir')
# ... training code ...
tf.profiler.experimental.stop()
```

Then view in TensorBoard:
```bash
tensorboard --logdir=logdir
```

## Troubleshooting

### Issue: "No GPU devices available"
**Solution**: Check CUDA/cuDNN installation:
```bash
nvidia-smi  # Check GPU is detected
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Issue: Out of memory errors
**Solutions**:
1. Reduce batch size (implicit in episode-based training)
2. Reduce `n_hidden_neurons`
3. Enable memory growth (see Performance Tips)

### Issue: Training is slow
**Solutions**:
1. Ensure GPU is being used
2. Profile to find bottlenecks
3. Consider using `@tf.function` on training loops (already implemented)

### Issue: Checkpoints not saving
**Solution**: Check directory permissions and disk space:
```bash
ls -la saved_models/
df -h
```

### Issue: Different results from TF1 version
**Note**: Minor numerical differences are expected due to:
- Different random number generation
- Different default initializations
- Different numerical precision in operations

For reproducible results, set random seeds:
```python
import numpy as np
import tensorflow as tf
import random

seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
```

## Parallel Processing

The main script uses `concurrent.futures.ProcessPoolExecutor` for parallel training:

```python
with concurrent.futures.ProcessPoolExecutor() as executor:
    results = executor.map(tf_function, ids)
```

To control parallelism:
```python
with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
    results = executor.map(tf_function, ids)
```

## Advanced: Distributed Training

For multi-GPU training, use TF2's distribution strategies:

```python
import tensorflow as tf

# Mirror strategy for multi-GPU
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    # Create model and optimizer inside strategy scope
    nnet = nn.neural_network(...)
    nnet.train()
```

## Getting Help

1. **Check documentation**: See `TF2_MIGRATION_README.md` for detailed migration info
2. **Test installation**: Run `test_tf2_conversion.py`
3. **Compare with TF1**: If TF1 version works but TF2 doesn't, compare configurations
4. **TensorFlow docs**: https://www.tensorflow.org/guide
5. **Issue tracking**: Check existing GitHub issues if this is a public repo

## Example Workflow

1. **Setup environment**
   ```bash
   pip install -r requirements.txt  # If available
   python test_tf2_conversion.py
   ```

2. **Configure training**
   - Edit `main_tf2.py` with desired hyperparameters

3. **Start training**
   ```bash
   python main_tf2.py
   ```

4. **Monitor progress**
   ```bash
   # In another terminal
   tensorboard --logdir=tensorboard/
   ```

5. **Analyze results**
   ```python
   import pickle
   import pandas as pd
   
   # Load test results
   with open('data/rnn_raw_data/{model_name}_test_...', 'rb') as f:
       df = pickle.load(f)
   
   # Analyze
   print(df.groupby('test_sd')['accuracy'].mean())
   ```

## Comparison with TF1 Version

| Feature | TF1 | TF2 |
|---------|-----|-----|
| Session Management | Required | Not needed |
| Eager Execution | Optional | Default |
| Model Definition | Functions + Placeholders | Keras Model class |
| Training Loop | `sess.run()` | `GradientTape` |
| Saving/Loading | `tf.train.Saver` | `tf.train.Checkpoint` |
| Debugging | Difficult | Easy (step-through) |
| Code Style | Verbose | Pythonic |
| Performance | Graph mode | Eager or `@tf.function` |

## Next Steps

- ✅ Complete: TF2 conversion
- 📊 Recommended: Run comparison between TF1 and TF2 results
- 🚀 Optional: Implement distributed training
- 📈 Optional: Add more comprehensive logging
- 🧪 Optional: Add unit tests for components

Happy training! 🎯
