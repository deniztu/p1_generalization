"""
TensorFlow 2 version of custom LSTM cell with optional noise injection.

This module implements a custom LSTM cell that can optionally add update-dependent
noise (Weber noise) to the hidden state, as described in Findling & Wyart, 2020.
"""

import tensorflow as tf
from tensorflow.keras import layers


class LSTM(layers.Layer):
    """
    Custom LSTM cell with optional noise injection.
    
    This LSTM cell implements the standard LSTM equations with an optional
    noise component that scales with the magnitude of state changes (Weber noise).
    
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
        Weight matrices for hidden-to-hidden connections [4, hidden_size, hidden_size].
        Indices: [0] input gate, [1] forget gate, [2] output gate, [3] gate weights.
    U : tf.Variable
        Weight matrices for input-to-hidden connections [4, input_size, hidden_size].
    b_i, b_f, b_o, b_g : tf.Variable
        Bias vectors for input, forget, output gates, and gate weights respectively.
    """
    
    def __init__(self, input_size, hidden_size, add_noise, **kwargs):
        """
        Initialize the LSTM cell.
        
        Parameters
        ----------
        input_size : int
            Size of the input vector.
        hidden_size : int
            Number of hidden units.
        add_noise : bool
            Whether to add update-dependent noise.
        """
        super(LSTM, self).__init__(**kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.add_noise = add_noise
        
        # Initialize weight matrices using Xavier/Glorot initialization
        initializer = tf.keras.initializers.GlorotUniform()
        
        # Weights for hidden-to-hidden connections (4 gates)
        self.W = self.add_weight(
            name='W',
            shape=[4, hidden_size, hidden_size],
            initializer=initializer,
            trainable=True
        )
        
        # Weights for input-to-hidden connections (4 gates)
        self.U = self.add_weight(
            name='U',
            shape=[4, input_size, hidden_size],
            initializer=initializer,
            trainable=True
        )
        
        # Bias terms for each gate
        self.b_i = self.add_weight(
            name='b_i',
            shape=[hidden_size],
            initializer=tf.keras.initializers.Zeros(),
            trainable=True
        )
        
        self.b_f = self.add_weight(
            name='b_f',
            shape=[hidden_size],
            initializer=tf.keras.initializers.Zeros(),
            trainable=True
        )
        
        self.b_o = self.add_weight(
            name='b_o',
            shape=[hidden_size],
            initializer=tf.keras.initializers.Zeros(),
            trainable=True
        )
        
        self.b_g = self.add_weight(
            name='b_g',
            shape=[hidden_size],
            initializer=tf.keras.initializers.Zeros(),
            trainable=True
        )
    
    def call(self, inputs, states, training=None):
        """
        Forward pass through the LSTM cell.
        
        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor. If add_noise is True, concatenated [input, noise].
            Shape: [batch_size, input_size + (hidden_size if add_noise else 0)]
        states : tuple of tf.Tensor
            Previous hidden state (h) and cell state (c).
            Each has shape: [batch_size, hidden_size]
        training : bool, optional
            Whether the layer is in training mode.
            
        Returns
        -------
        outputs : tf.Tensor
            New hidden state. Shape: [batch_size, hidden_size]
        new_states : tuple of tf.Tensor
            New (h, c) states.
        """
        ht_1, ct_1 = states
        
        # Separate input from noise if noise is being added
        if self.add_noise:
            x = inputs[:, :self.input_size]
            noise_h = inputs[:, -self.hidden_size:]
        else:
            x = inputs
        
        # LSTM gates computation
        # Input gate
        i = tf.sigmoid(
            tf.matmul(x, self.U[0]) + tf.matmul(ht_1, self.W[0]) + self.b_i
        )
        
        # Forget gate
        f = tf.sigmoid(
            tf.matmul(x, self.U[1]) + tf.matmul(ht_1, self.W[1]) + self.b_f
        )
        
        # Output gate
        o = tf.sigmoid(
            tf.matmul(x, self.U[2]) + tf.matmul(ht_1, self.W[2]) + self.b_o
        )
        
        # Gate weights (candidate cell state)
        g = tf.tanh(
            tf.matmul(x, self.U[3]) + tf.matmul(ht_1, self.W[3]) + self.b_g
        )
        
        # New cell state
        ct = ct_1 * f + g * i
        
        # New hidden state (before noise)
        ht = tf.tanh(ct) * o
        ht_exact = ht
        
        # Add update-dependent noise if enabled
        if self.add_noise:
            # Noise scales with the magnitude of the state change (Weber noise)
            noise_added = noise_h * tf.abs(ht - ht_1)
            
            # Stop gradient to treat noise as a constraint
            noise_added = tf.stop_gradient(noise_added)
            
            # Add noise to hidden state
            ht = ht + noise_added
        
        return ht, (ht, ct)
    
    def get_initial_state(self, batch_size=None):
        """
        Get initial zero states for the LSTM.
        
        Parameters
        ----------
        batch_size : int, optional
            Batch size. If None, returns states for batch_size=1.
            
        Returns
        -------
        tuple of tf.Tensor
            Initial (h, c) states, each of shape [batch_size, hidden_size].
        """
        if batch_size is None:
            batch_size = 1
            
        h_init = tf.zeros([batch_size, self.hidden_size], dtype=tf.float32)
        c_init = tf.zeros([batch_size, self.hidden_size], dtype=tf.float32)
        
        return (h_init, c_init)


class LSTMLayer(layers.Layer):
    """
    Wrapper layer for custom LSTM cell to work with sequences.
    
    This layer wraps the custom LSTM cell and processes sequences of inputs,
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
        super(LSTMLayer, self).__init__(**kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.add_noise = add_noise
        self.return_sequences = return_sequences
        self.cell = LSTM(input_size, hidden_size, add_noise)
    
    def call(self, inputs, initial_state=None, training=None):
        """
        Process a sequence of inputs through the LSTM.
        
        Parameters
        ----------
        inputs : tf.Tensor
            Input sequence. Shape: [batch_size, time_steps, features]
        initial_state : tuple of tf.Tensor, optional
            Initial (h, c) states. If None, uses zeros.
        training : bool, optional
            Whether in training mode.
            
        Returns
        -------
        outputs : tf.Tensor
            If return_sequences=True: [batch_size, time_steps, hidden_size]
            If return_sequences=False: [batch_size, hidden_size]
        final_state : tuple of tf.Tensor
            Final (h, c) states.
        """
        batch_size = tf.shape(inputs)[0]
        
        if initial_state is None:
            states = self.cell.get_initial_state(batch_size)
        else:
            states = initial_state
        
        # Process sequence using tf.scan or manual loop
        # For simplicity, using manual loop (can be optimized with tf.scan)
        outputs = []
        for t in range(inputs.shape[1]):
            output, states = self.cell(inputs[:, t, :], states, training=training)
            outputs.append(output)
        
        if self.return_sequences:
            outputs = tf.stack(outputs, axis=1)
        else:
            outputs = outputs[-1]
        
        return outputs, states
