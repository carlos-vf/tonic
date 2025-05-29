import tensorflow as tf
import keras

@keras.saving.register_keras_serializable()
class ObservationEncoder(tf.keras.Model):
    def __init__(self, observation_normalizer=None, **kwargs): # Added __init__ and **kwargs
        super().__init__(**kwargs)
        # Store the normalizer. If it's a Keras layer, it will be tracked automatically.
        self.observation_normalizer = observation_normalizer

    def initialize(self, observation_normalizer=None):
        # This method might be called by tonic for runtime setup.
        # Ensure that if a normalizer is passed here, it updates the instance.
        if observation_normalizer is not None:
            self.observation_normalizer = observation_normalizer

    def call(self, observations):
        if self.observation_normalizer:
            observations = self.observation_normalizer(observations)
        return observations

    def get_config(self):
        # Start with the base config from tf.keras.Model
        config = super().get_config()
        
        # Add the configuration of the observation_normalizer if it exists
        if self.observation_normalizer:
            config['observation_normalizer'] = keras.saving.serialize_keras_object(self.observation_normalizer)
        else:
            # Explicitly store None if no normalizer is set, for consistent deserialization
            config['observation_normalizer'] = None 
        
        return config

    @classmethod
    def from_config(cls, config):
        # Extract the normalizer's configuration. Use .pop with a default of None.
        normalizer_config = config.pop('observation_normalizer', None)
        
        # Deserialize the normalizer if its config exists
        observation_normalizer = None
        if normalizer_config:
            observation_normalizer = keras.saving.deserialize_keras_object(normalizer_config)
        
        # Create an instance of ObservationEncoder using the deserialized normalizer
        # and any remaining standard Keras arguments (**config)
        return cls(observation_normalizer=observation_normalizer, **config)


@keras.saving.register_keras_serializable()
class ObservationActionEncoder(tf.keras.Model):
    def initialize(self, observation_normalizer=None):
        self.observation_normalizer = observation_normalizer

    def call(self, observations, actions):
        if self.observation_normalizer:
            observations = self.observation_normalizer(observations)
        return tf.concat([observations, actions], axis=-1)
