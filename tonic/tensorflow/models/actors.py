import tensorflow as tf
import tensorflow_probability as tfp
import keras 

from tonic.tensorflow import models


FLOAT_EPSILON = 1e-8


class SquashedMultivariateNormalDiag:
    def __init__(self, loc, scale):
        self._distribution = tfp.distributions.MultivariateNormalDiag(
            loc, scale)

    def sample_with_log_prob(self, shape=()):
        samples = self._distribution.sample(shape)
        squashed_samples = tf.tanh(samples)
        log_probs = self._distribution.log_prob(samples)
        log_probs -= tf.reduce_sum(
            tf.math.log(1 - squashed_samples ** 2 + 1e-6), axis=-1)
        return squashed_samples, log_probs

    def sample(self, shape=()):
        samples = self._distribution.sample(shape)
        return tf.tanh(samples)

    def log_prob(self, samples):
        '''Required unsquashed samples cannot be accurately recovered.'''
        raise NotImplementedError(
            'Not implemented to avoid approximation errors. '
            'Use sample_with_log_prob directly.')

    def mode(self):
        return tf.tanh(self._distribution.mode())


class DetachedScaleGaussianPolicyHead(tf.keras.Model):
    def __init__(
        self, loc_activation='tanh', dense_loc_kwargs=None, log_scale_init=0.,
        scale_min=1e-4, scale_max=1.,
        distribution=tfp.distributions.MultivariateNormalDiag
    ):
        super().__init__()
        self.loc_activation = loc_activation
        if dense_loc_kwargs is None:
            dense_loc_kwargs = models.default_dense_kwargs()
        self.dense_loc_kwargs = dense_loc_kwargs
        self.log_scale_init = log_scale_init
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.distribution = distribution

    def initialize(self, action_size):
        self.loc_layer = tf.keras.layers.Dense(
            action_size, self.loc_activation, **self.dense_loc_kwargs)
        log_scale = [[self.log_scale_init] * action_size]
        self.log_scale = tf.Variable(log_scale, dtype=tf.float32)

    def call(self, inputs):
        loc = self.loc_layer(inputs)
        batch_size = tf.shape(inputs)[0]
        scale = tf.math.softplus(self.log_scale) + FLOAT_EPSILON
        scale = tf.clip_by_value(scale, self.scale_min, self.scale_max)
        scale = tf.tile(scale, (batch_size, 1))
        return self.distribution(loc, scale)


class GaussianPolicyHead(tf.keras.Model):
    def __init__(
        self, loc_activation='tanh', dense_loc_kwargs=None,
        scale_activation='softplus', scale_min=1e-4, scale_max=1,
        dense_scale_kwargs=None,
        distribution=tfp.distributions.MultivariateNormalDiag
    ):
        super().__init__()
        self.loc_activation = loc_activation
        if dense_loc_kwargs is None:
            dense_loc_kwargs = models.default_dense_kwargs()
        self.dense_loc_kwargs = dense_loc_kwargs
        self.scale_activation = scale_activation
        self.scale_min = scale_min
        self.scale_max = scale_max
        if dense_scale_kwargs is None:
            dense_scale_kwargs = models.default_dense_kwargs()
        self.dense_scale_kwargs = dense_scale_kwargs
        self.distribution = distribution

    def initialize(self, action_size):
        self.loc_layer = tf.keras.layers.Dense(
            action_size, self.loc_activation, **self.dense_loc_kwargs)
        self.scale_layer = tf.keras.layers.Dense(
            action_size, self.scale_activation, **self.dense_scale_kwargs)

    def call(self, inputs):
        loc = self.loc_layer(inputs)
        scale = self.scale_layer(inputs)
        scale = tf.clip_by_value(scale, self.scale_min, self.scale_max)
        return self.distribution(loc, scale)


@keras.saving.register_keras_serializable()
class DeterministicPolicyHead(tf.keras.Model):
    def __init__(self, action_size=None, activation='tanh', dense_kwargs=None, **kwargs): # Added action_size and **kwargs
        super().__init__(**kwargs) # Pass **kwargs to the parent constructor

        # Store these for get_config
        self._action_size = action_size # Store action_size
        self._activation = activation
        
        if dense_kwargs is None:
            dense_kwargs = models.default_dense_kwargs()
        self._dense_kwargs = dense_kwargs # Store dense_kwargs

        # Initialize action_layer here if action_size is known at init time
        # This is more robust for Keras serialization.
        if self._action_size is not None:
            self.action_layer = tf.keras.layers.Dense(
                self._action_size, self._activation, **self._dense_kwargs)
        else:
            self.action_layer = None # It will be created in initialize if needed

    def initialize(self, action_size):
        # Update action_size and create layer if not already done in __init__
        if self.action_layer is None or self._action_size != action_size:
            self._action_size = action_size
            self.action_layer = tf.keras.layers.Dense(
                self._action_size, self._activation, **self._dense_kwargs)

    def call(self, inputs):
        # Ensure action_layer is built before calling
        if self.action_layer is None:
            raise ValueError("action_layer not initialized. Call .initialize(action_size) first or provide action_size during instantiation.")
        return self.action_layer(inputs)

    def get_config(self):
        # Start with the base config from tf.keras.Model
        config = super().get_config()
        
        # Add the specific arguments for this class's __init__
        config['action_size'] = self._action_size
        config['activation'] = self._activation
        config['dense_kwargs'] = self._dense_kwargs # Store the actual dict

        # If self.action_layer was a custom Keras object (which tf.keras.layers.Dense is not),
        # you'd serialize it using keras.saving.serialize_keras_object(self.action_layer).
        # For standard Keras layers like Dense, they are automatically handled as part of the model.
        
        return config


@keras.saving.register_keras_serializable()
class Actor(tf.keras.Model):
    def __init__(self, encoder, torso, head, **kwargs): # Added **kwargs
        super().__init__(**kwargs) # Passed **kwargs to the parent constructor
        self.encoder = encoder
        self.torso = torso
        self.head = head

    def initialize(
        self, observation_space, action_space, observation_normalizer=None
    ):
        self.encoder.initialize(observation_normalizer)
        self.head.initialize(action_space.shape[0])

    def call(self, *inputs):
        out = self.encoder(*inputs)
        out = self.torso(out)
        return self.head(out)

    def get_config(self):
        # Start with the base config from tf.keras.Model
        config = super().get_config()
        
        # Add the configurations of your sub-models/layers
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
        
        # Create an instance of Actor using the deserialized components
        # Pass the remaining config (like 'name', 'trainable', 'dtype') to the constructor
        return cls(encoder=encoder, torso=torso, head=head, **config)
