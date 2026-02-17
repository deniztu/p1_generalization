# TensorFlow 2 Migration

This directory contains the TensorFlow 2 versions of the neural network training code.

### Main Scripts
- **`main_tf2.py`** - TensorFlow 2 version of the main training script
  - Uses the new TF2 neural network class
  - Maintains the same functionality as the original `main.py`
  - No `tf.Session()` context managers needed
  - Uses eager execution by default

### Neural Network Classes
- **`classes/neural_networks/p_network_class_organized_lstm_cell_tf2.py`** - Complete TF2 rewrite of the Actor-Critic network
  - Inherits from `tf.keras.Model`
  - Uses `tf.GradientTape` for training
  - Uses `tf.train.Checkpoint` for saving/loading
  - Implements custom training loops with eager execution
  - Uses `tf.summary` for TensorBoard logging

### Custom RNN Cells
- **`classes/neural_networks/rnns/own_lstm_cell_tf2.py`** - TF2 version of custom LSTM with noise injection
  - Inherits from `tf.keras.layers.Layer`
  - Implements Weber noise (update-dependent noise)
  - Compatible with TF2 eager execution

- **`classes/neural_networks/rnns/recurrent_networks_tf2.py`** - TF2 version of custom Elman RNN with noise
  - Inherits from `tf.keras.layers.Layer`
  - Implements update-dependent noise for RNN
  - Compatible with TF2 eager execution

## Key Changes from TensorFlow 1 to TensorFlow 2

### 1. **Session Management**
- **TF1**: Required `tf.Session()` context manager and `sess.run()`
- **TF2**: Eager execution by default, no sessions needed

### 2. **Placeholders → Function Arguments**
- **TF1**: `self.prev_rewardsch = tf.placeholder(shape=[None,1], dtype=tf.float32)`
- **TF2**: Pass as function arguments directly to `model(inputs)`

### 3. **RNN Cells**
- **TF1**: `tf.contrib.rnn.BasicLSTMCell(nb_units, state_is_tuple=True)`
- **TF2**: `tf.keras.layers.LSTMCell(nb_units)` or custom layers inheriting from `tf.keras.layers.Layer`

### 4. **Network Architecture**
- **TF1**: Used `tensorflow.contrib.slim` for layers
- **TF2**: Inherit from `tf.keras.Model` and use `tf.keras.layers`

### 5. **Training Loop**
- **TF1**: 
  ```python
  sess.run([self.apply_grads], feed_dict=feed_dict)
  ```
- **TF2**: 
  ```python
  with tf.GradientTape() as tape:
      loss = compute_loss(...)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  ```

### 6. **Optimizers**
- **TF1**: `tf.train.RMSPropOptimizer(learning_rate=lr)`
- **TF2**: `tf.keras.optimizers.RMSprop(learning_rate=lr)`

### 7. **Saving/Loading Models**
- **TF1**: 
  ```python
  saver = tf.train.Saver()
  saver.save(sess, path)
  saver.restore(sess, path)
  ```
- **TF2**: 
  ```python
  checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
  checkpoint.save(path)
  checkpoint.restore(path)
  ```

### 8. **TensorBoard Logging**
- **TF1**: `tf.summary.FileWriter(path)`
- **TF2**: 
  ```python
  summary_writer = tf.summary.create_file_writer(path)
  with summary_writer.as_default():
      tf.summary.scalar('loss', loss, step=epoch)
  ```

### 9. **Graph Reset**
- **TF1**: `tf.reset_default_graph()`
- **TF2**: Not needed (eager execution mode)

### 10. **Variable Initialization**
- **TF1**: `sess.run(tf.global_variables_initializer())`
- **TF2**: Variables are initialized automatically when created

## Usage

### Training
```python
python main_tf2.py
```

The script will:
1. Create RNN instances with different hyperparameters
2. Train them on bandit tasks using parallel processing
3. Save checkpoints and TensorBoard logs
4. Test on specified bandit configurations

### Configuration
Edit the global variables in `main_tf2.py`:
```python
N_HIDDEN = [48, 64, 80]  # Number of hidden units to test
ENTROPIES = [0, 0.05, 'linear']  # Entropy loss weights
IDS = [0, 1, 2, 3]  # RNN instance IDs
TRAIN_RNN = True  # Whether to train
TEST_DAW = False  # Whether to test on Daw walks
```

## Compatibility

- **Python**: 3.7+
- **TensorFlow**: 2.0+
- **NumPy**, **Pandas**, **SciPy**: Same versions as TF1 code

## Performance Considerations

### TF2 Advantages
1. **Easier Debugging**: Eager execution allows step-through debugging
2. **Cleaner Code**: No session management, more Pythonic
3. **Better API**: Keras integration provides high-level APIs
4. **@tf.function**: Decorator for graph compilation when needed

### Optimization Tips
1. Use `@tf.function` decorator on the `compute_loss` method for graph compilation
2. Consider `tf.data.Dataset` for data loading if working with larger datasets
3. Enable mixed precision training for faster computation on modern GPUs:
   ```python
   from tensorflow.keras import mixed_precision
   policy = mixed_precision.Policy('mixed_float16')
   mixed_precision.set_global_policy(policy)
   ```

## Testing

Both TF1 and TF2 versions should produce equivalent results. You can verify by:
1. Training the same configuration with both versions
2. Comparing final performance metrics
3. Comparing saved model outputs on the same test data

## Notes

- The `reset()` method in `neural_network` class is kept for API compatibility but does nothing in TF2 (no-op)
- Custom RNN cells maintain the same noise injection behavior as the TF1 version
- All helper functions from the original code (`discount`, `normalized_columns_initializer`, etc.) are preserved with TF2-compatible implementations

## Troubleshooting

### Common Issues

1. **CUDA/cuDNN version mismatch**: Ensure your CUDA and cuDNN versions are compatible with your TensorFlow 2 installation
   
2. **Memory issues**: TF2's eager execution can be more memory-intensive. Try:
   ```python
   # Limit GPU memory growth
   gpus = tf.config.experimental.list_physical_devices('GPU')
   for gpu in gpus:
       tf.config.experimental.set_memory_growth(gpu, True)
   ```

3. **Slower than TF1**: Wrap training loops with `@tf.function` for graph compilation:
   ```python
   @tf.function
   def train_step(...):
       # training code
   ```

## Future Improvements

- [ ] Add distributed training support using `tf.distribute.Strategy`
- [ ] Implement data pipeline using `tf.data.Dataset`
- [ ] Add mixed precision training support
- [ ] Create model serving exports using SavedModel format
- [ ] Add unit tests for TF2 implementation

## References

- [TensorFlow 2 Migration Guide](https://www.tensorflow.org/guide/migrate)
- [Keras API Documentation](https://www.tensorflow.org/api_docs/python/tf/keras)
- [Custom Layers and Models](https://www.tensorflow.org/guide/keras/custom_layers_and_models)
- Findling, C., & Wyart, V. (2020). Computation noise in human learning and decision-making. *bioRxiv*.
