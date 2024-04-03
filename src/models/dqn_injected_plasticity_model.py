import tensorflow as tf

from src.models.layers.max_norm import MaxNorm


@tf.keras.utils.register_keras_serializable(package="src.models")
class DQNInjectedPlasticityModel(tf.keras.Model):
    def __init__(self, num_classes, seed, input_shape, eta, alpha, **kwargs):
        super(DQNInjectedPlasticityModel, self).__init__(**kwargs)

        self.seed = seed
        if seed is not None:
            tf.random.set_seed(seed)

        self.num_classes = num_classes
        self.input_shape = input_shape
        self.eta = eta
        self.alpha = alpha

        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=8,
            strides=4,
            activation="relu",
            input_shape=input_shape
        )
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, activation="relu")
        self.conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation="relu")
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=512, activation="relu")
        self.state_values = tf.keras.layers.Dense(units=1)
        self.raw_advantages = tf.keras.layers.Dense(num_classes)
        self.advantages = MaxNorm()
        self.q_values = tf.keras.layers.Add()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)

        state_values = self.state_values(x)
        raw_advantages = self.raw_advantages(x)

        advantages = self.advantages(raw_advantages)
        q_values = self.q_values([state_values, advantages])
        return q_values

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'seed': self.seed,
            'input_shape': self.input_shape,
            'eta': self.eta,
            'alpha': self.alpha
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            num_classes=config['num_classes'],
            seed=config['seed'],
            input_shape=config['input_shape'],
            eta=config['eta'],
            alpha=config['alpha']
        )

