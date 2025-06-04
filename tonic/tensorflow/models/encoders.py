import tensorflow as tf
import keras

@keras.saving.register_keras_serializable()
class ObservationEncoder(tf.keras.Model):
    def __init__(self, observation_normalizer=None, **kwargs):
        super().__init__(**kwargs)
        self.observation_normalizer = observation_normalizer

    def initialize(self, observation_normalizer=None):
        if observation_normalizer is not None:
            self.observation_normalizer = observation_normalizer

    def call(self, observations):
        if self.observation_normalizer:
            observations = self.observation_normalizer(observations)
        return observations

    def get_config(self):
        config = super().get_config()
        if self.observation_normalizer:
            config['observation_normalizer'] = keras.saving.serialize_keras_object(self.observation_normalizer)
        else:
            config['observation_normalizer'] = None 
        
        return config

    @classmethod
    def from_config(cls, config):
        normalizer_config = config.pop('observation_normalizer', None)
        observation_normalizer = None
        if normalizer_config:
            observation_normalizer = keras.saving.deserialize_keras_object(normalizer_config)
        return cls(observation_normalizer=observation_normalizer, **config)


@keras.saving.register_keras_serializable()
class ObservationActionEncoder(tf.keras.Model):
    def initialize(self, observation_normalizer=None):
        self.observation_normalizer = observation_normalizer

    def call(self, observations, actions):
        if self.observation_normalizer:
            observations = self.observation_normalizer(observations)
        return tf.concat([observations, actions], axis=-1)
