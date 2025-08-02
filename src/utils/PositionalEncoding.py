# utils/PositionalEncoding.py

import tensorflow as tf
from keras import layers
import numpy as np
class PositionalEncoding(layers.Layer):
    def __init__(self, sequence_length, d_model, **kwargs):
        # Pass any extra kwargs like "trainable", "name", "dtype" to the Layer base class.
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.d_model = d_model
        
        # Create your pos_encoding here or in build().
        # For example:
        import numpy as np
        position = np.arange(self.sequence_length)[:, np.newaxis]  # (seq_len, 1)
        div_term = np.exp(np.arange(0, self.d_model, 2)
                          * -(np.log(10000.0) / self.d_model))
        pos_encoding = np.zeros((self.sequence_length, self.d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        self.pos_encoding = tf.constant(pos_encoding[np.newaxis, ...], dtype=tf.float32)

    def call(self, x):
        # x shape: (batch_size, seq_length, d_model)
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]

    def get_config(self):
        # Let Keras serialize your custom arguments
        config = super().get_config()  # This already has 'trainable', 'name', etc.
        config.update({
            "sequence_length": self.sequence_length,
            "d_model": self.d_model
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class PositionalEncodingForSpectral(layers.Layer):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncodingForSpectral, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Compute the positional encodings once in log space.
        position = np.arange(self.max_len)[:, np.newaxis]  # (max_len, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pos_encoding = np.zeros((self.max_len, d_model))
        
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        
        if d_model % 2 == 0:
            pos_encoding[:, 1::2] = np.cos(position * div_term)
        else:
            pos_encoding[:, 1::2] = np.cos(position * div_term[:self.d_model//2])
        
        pos_encoding = pos_encoding[np.newaxis, ...]  # Shape: (1, max_len, d_model)
        self.pos_encoding = tf.constant(pos_encoding, dtype=tf.float32)
    
    def call(self, x):
        # x shape: (batch_size, seq_length, d_model)
        seq_length = tf.shape(x)[1]
        # Ensure that seq_length does not exceed max_len by relying on model design
        # Optionally, you can use tf.debugging.assert_less_equal
        # but it's optional
        return x + self.pos_encoding[:, :seq_length, :]
    
    def compute_output_shape(self, input_shape):
        return input_shape






# class PositionalEncoding(Layer):
#     def __init__(self, sequence_length, d_model):
#         super(PositionalEncoding, self).__init__()
#         self.sequence_length = sequence_length
#         self.d_model = d_model
#         # Compute the positional encodings once in log space.
#         position = np.arange(self.sequence_length)[:, np.newaxis]
#         div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
#         pos_encoding = np.zeros((self.sequence_length, self.d_model))
#         pos_encoding[:, 0::2] = np.sin(position * div_term)
#         pos_encoding[:, 1::2] = np.cos(position * div_term)
#         pos_encoding = pos_encoding[np.newaxis, ...]  # Shape: (1, sequence_length, d_model)
#         self.pos_encoding = tf.constant(pos_encoding, dtype=tf.float32)

#     def call(self, x):
#         # x shape: (batch_size, sequence_length, d_model)
#         return x + self.pos_encoding[:, :tf.shape(x)[1], :]
