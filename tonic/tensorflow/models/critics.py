import tensorflow as tf
import keras

from tonic.tensorflow import models


class ValueHead(tf.keras.Model):
    def __init__(self, dense_kwargs=None):
        super().__init__()
        if dense_kwargs is None:
            dense_kwargs = models.default_dense_kwargs()
        self.v_layer = tf.keras.layers.Dense(1, **dense_kwargs)

    def initialize(self, return_normalizer=None):
        self.return_normalizer = return_normalizer

    def call(self, inputs):
        out = self.v_layer(inputs)
        out = tf.squeeze(out, -1)
        if self.return_normalizer:
            out = self.return_normalizer(out)
        return out


class CategoricalWithSupport:
    def __init__(self, values, logits):
        self.values = values
        self.logits = logits
        self.probabilities = tf.nn.softmax(logits)

    def mean(self):
        return tf.reduce_sum(self.probabilities * self.values, axis=-1)

    def project(self, returns):
        vmin, vmax = self.values[0], self.values[-1]
        d_pos = tf.concat([self.values, vmin[None]], 0)[1:]
        d_pos = (d_pos - self.values)[None, :, None]
        d_neg = tf.concat([vmax[None], self.values], 0)[:-1]
        d_neg = (self.values - d_neg)[None, :, None]

        clipped_returns = tf.clip_by_value(returns, vmin, vmax)
        delta_values = clipped_returns[:, None] - self.values[None, :, None]
        delta_sign = tf.cast(delta_values >= 0, tf.float32)
        delta_hat = ((delta_sign * delta_values / d_pos) -
                     ((1 - delta_sign) * delta_values / d_neg))
        delta_clipped = tf.clip_by_value(1 - delta_hat, 0, 1)

        return tf.reduce_sum(delta_clipped * self.probabilities[:, None], 2)


@keras.saving.register_keras_serializable()
class DistributionalValueHead(tf.keras.Model):
    def __init__(self, vmin, vmax, num_atoms, dense_kwargs=None, **kwargs): # Added **kwargs
        super().__init__(**kwargs) # Passed **kwargs to the parent constructor

        if dense_kwargs is None:
            dense_kwargs = models.default_dense_kwargs()
        
        # Store these for get_config
        self._vmin = vmin
        self._vmax = vmax
        self._num_atoms = num_atoms
        self._dense_kwargs = dense_kwargs

        self.distributional_layer = tf.keras.layers.Dense(
            num_atoms, **dense_kwargs)
        self.values = tf.cast(tf.linspace(vmin, vmax, num_atoms), tf.float32)

    def initialize(self, return_normalizer=None):
        if return_normalizer:
            raise ValueError(
                'Return normalizers cannot be used with distributional value'
                'heads.')

    def call(self, inputs):
        logits = self.distributional_layer(inputs)
        return CategoricalWithSupport(values=self.values, logits=logits)

    def get_config(self):
        # Start with the base config from tf.keras.Model
        config = super().get_config()
        
        # Add the specific arguments for this class's __init__
        config['vmin'] = self._vmin
        config['vmax'] = self._vmax
        config['num_atoms'] = self._num_atoms
        # Only serialize dense_kwargs if it's not the default, or if it's a serializable dict
        if self._dense_kwargs is not None:
             config['dense_kwargs'] = self._dense_kwargs
        
        # If distributional_layer was a custom Keras object (which Dense is not),
        # you'd serialize it similarly to how you did in Critic's get_config.
        # For standard Keras layers like Dense, they are automatically handled.
        
        return config

    @classmethod
    def from_config(cls, config):
        # Extract the specific arguments for this class's __init__
        vmin = config.pop('vmin')
        vmax = config.pop('vmax')
        num_atoms = config.pop('num_atoms')
        dense_kwargs = config.pop('dense_kwargs', None) # Use .pop with default for optional args

        # Create an instance of DistributionalValueHead using the extracted arguments
        # Pass the remaining config (like 'name', 'trainable', 'dtype') to the constructor
        return cls(vmin=vmin, vmax=vmax, num_atoms=num_atoms, dense_kwargs=dense_kwargs, **config)


@keras.saving.register_keras_serializable()
class Critic(tf.keras.Model):
    def __init__(self, encoder, torso, head, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.torso = torso
        self.head = head

    def initialize(
        self, observation_space, action_space, observation_normalizer=None,
        return_normalizer=None
    ):
        self.encoder.initialize(observation_normalizer)
        self.head.initialize(return_normalizer)

    def call(self, *inputs):
        out = self.encoder(*inputs)
        out = self.torso(out)
        return self.head(out)

    def get_config(self):
        # Start with the base config from tf.keras.Model
        config = super().get_config()
        
        # Add the configurations of your sub-models/layers
        # Keras will automatically handle serialization for Keras objects
        # if they are registered or standard Keras layers.
        config['encoder'] = keras.saving.serialize_keras_object(self.encoder)
        config['torso'] = keras.saving.serialize_keras_object(self.torso)
        config['head'] = keras.saving.serialize_keras_object(self.head)
        
        return config

    @classmethod
    def from_config(cls, config):
        # Extract the configurations of your sub-models/layers
        encoder_config = config.pop('encoder')
        torso_config = config.pop('torso')
        head_config = config.pop('head')

        # Deserialize the sub-models/layers
        encoder = keras.saving.deserialize_keras_object(encoder_config)
        torso = keras.saving.deserialize_keras_object(torso_config)
        head = keras.saving.deserialize_keras_object(head_config)
        
        # Create an instance of Critic using the deserialized components
        # Pass the remaining config (like 'name', 'trainable', 'dtype') to the constructor
        return cls(encoder=encoder, torso=torso, head=head, **config)