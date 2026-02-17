"""
TensorFlow 2 version of custom RNN cell with optional noise injection.

This module implements a custom Elman RNN cell that can optionally add
update-dependent noise (Weber noise) to the hidden state.
"""

import tensorflow as tf
from tensorflow.keras import layers


class RNN(layers.Layer):
    """
    Custom Elman RNN cell with optional noise injection.
    
    Implements a simple recurrent neural network with update-dependent noise
    that scales with the magnitude of state changes (Weber noise).
    
    Parameters
    ----------
    input_size : int
        Size of the input vector (excluding noise).
    hidden_size : int
        Number of units in the hidden state.
    add_noise : bool
        If True, adds update-dependent noise to the hidden state.
        
    Attributes
    ----------
    W : tf.Variable
        Weight matrix for hidden-to-hidden connections [hidden_size, hidden_size].
    U : tf.Variable
        Weight matrix for input-to-hidden connections [input_size, hidden_size].
    b : tf.Variable
        Bias vector for the hidden state.
    """
    
    def __init__(self, input_size, hidden_size, add_noise, **kwargs):
        """
        Initialize the RNN cell.
        
        Parameters
        ----------
        input_size : int
            Size of the input vector.
        hidden_size : int
            Number of hidden units.
        add_noise : bool
            Whether to add update-dependent noise.
        """
        super(RNN, self).__init__(**kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.add_noise = add_noise
        
        # Initialize weight matrices using Xavier/Glorot initialization
        initializer = tf.keras.initializers.GlorotUniform()
        
        # Weight matrix for hidden-to-hidden connections
        self.W = self.add_weight(
            name='W',
            shape=[hidden_size, hidden_size],
            initializer=initializer,
            trainable=True
        )
        
        # Weight matrix for input-to-hidden connections
        self.U = self.add_weight(
            name='U',
            shape=[input_size, hidden_size],
            initializer=initializer,
            trainable=True
        )
        
        # Bias term
        self.b = self.add_weight(
            name='b',
            shape=[hidden_size],
            initializer=tf.keras.initializers.Zeros(),
            trainable=True
        )
    
    def call(self, inputs, states, training=None):
        """
        Forward pass through the RNN cell.
        
        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor. If add_noise is True, concatenated [input, noise].
            Shape: [batch_size, input_size + (hidden_size if add_noise else 0)]
        states : tuple
            Previous hidden state (st_1,) and noise statistics.
            st_1 has shape: [batch_size, hidden_size]
        training : bool, optional
            Whether the layer is in training mode.
            
        Returns
        -------
        outputs : tf.Tensor
            New hidden state. Shape: [batch_size, hidden_size]
        new_states : tuple
            New state and noise mean.
        """
        st_1, noise_mean_prev = states
        
        # Separate input from noise if noise is being added
        if self.add_noise:
            x = inputs[:, :self.input_size]
            noise_h = inputs[:, -self.hidden_size:]
        else:
            x = inputs
        
        # Update hidden state: h_t = tanh(W*h_{t-1} + U*x_t + b)
        ht = tf.matmul(st_1, self.W) + tf.matmul(x, self.U) + self.b
        ht_exact = ht
        
        # Add update-dependent noise if enabled
        if self.add_noise:
            # Noise scales with the magnitude of the state change (Weber noise)
            noise_added = noise_h * tf.abs(ht - st_1)
            
            # Stop gradient to treat noise as a constraint
            noise_added = tf.stop_gradient(noise_added)
            
            # Add noise to hidden state
            ht = ht + noise_added
            
            # Calculate mean absolute noise
            noise_mean = tf.reduce_mean(tf.abs(ht - ht_exact))
        else:
            noise_mean = 0.0
        
        # Apply tanh activation
        ht = tf.tanh(ht)
        
        return ht, (ht, noise_mean)
    
    def get_initial_state(self, batch_size=None):
        """
        Get initial zero state for the RNN.
        
        Parameters
        ----------
        batch_size : int, optional
            Batch size. If None, returns state for batch_size=1.
            
        Returns
        -------
        tuple
            Initial state (h, noise_mean), where h has shape [batch_size, hidden_size].
        """
        if batch_size is None:
            batch_size = 1
            
        h_init = tf.zeros([batch_size, self.hidden_size], dtype=tf.float32)
        noise_mean_init = 0.0
        
        return (h_init, noise_mean_init)


class RNNLayer(layers.Layer):
    """
    Wrapper layer for custom RNN cell to work with sequences.
    
    This layer wraps the custom RNN cell and processes sequences of inputs,
    maintaining state across time steps.
    
    Parameters
    ----------
    input_size : int
        Size of input features.
    hidden_size : int
        Number of hidden units.
    add_noise : bool
        Whether to add update-dependent noise.
    return_sequences : bool, optional
        If True, returns hidden states for all time steps.
        If False, returns only the final hidden state.
    """
    
    def __init__(self, input_size, hidden_size, add_noise, return_sequences=True, **kwargs):
        super(RNNLayer, self).__init__(**kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.add_noise = add_noise
        self.return_sequences = return_sequences
        self.cell = RNN(input_size, hidden_size, add_noise)
    
    def call(self, inputs, initial_state=None, training=None):
        """
        Process a sequence of inputs through the RNN.
        
        Parameters
        ----------
        inputs : tf.Tensor
            Input sequence. Shape: [batch_size, time_steps, features]
        initial_state : tuple, optional
            Initial (h, noise_mean). If None, uses zeros.
        training : bool, optional
            Whether in training mode.
            
        Returns
        -------
        outputs : tf.Tensor
            If return_sequences=True: [batch_size, time_steps, hidden_size]
            If return_sequences=False: [batch_size, hidden_size]
        final_state : tuple
            Final (h, noise_mean) state.
        noise_means : list
            List of noise means for each time step.
        """
        batch_size = tf.shape(inputs)[0]
        
        if initial_state is None:
            states = self.cell.get_initial_state(batch_size)
        else:
            states = initial_state
        
        # Process sequence
        outputs = []
        noise_means = []
        
        for t in range(inputs.shape[1]):
            output, states = self.cell(inputs[:, t, :], states, training=training)
            outputs.append(output)
            noise_means.append(states[1])
        
        if self.return_sequences:
            outputs = tf.stack(outputs, axis=1)
        else:
            outputs = outputs[-1]
        
        return outputs, states, noise_means
