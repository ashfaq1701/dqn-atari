import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="src.models")
class DQNInjectedPlasticityModel(tf.keras.Model):
    def __init__(
            self,
            num_classes,
            seed,
            input_shape,
            eta,
            alpha,
            is_plasticity_injected=False,
            hebb1=None,
            hebb2=None,
            **kwargs
    ):
        super(DQNInjectedPlasticityModel, self).__init__(**kwargs)

        self.seed = seed
        if seed is not None:
            tf.random.set_seed(seed)

        self.num_classes = num_classes
        self.input_shape = input_shape
        self.eta = eta
        self.alpha = alpha
        self.is_plasticity_injected = is_plasticity_injected

        if hebb1 is None:
            self.hebb1 = tf.Variable(tf.zeros(shape=(512, 256)), trainable=False)
        else:
            self.hebb1 = hebb1

        if hebb2 is None:
            self.hebb2 = tf.Variable(tf.zeros(shape=(256, num_classes + 1)), trainable=False)
        else:
            self.hebb2 = hebb2

        self.hebb2_adv = tf.Variable(self.hebb2[:, :num_classes], trainable=False)
        self.hebb2_states = tf.Variable(self.hebb2[:, num_classes:], trainable=False)

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
        self.dense2 = tf.keras.layers.Dense(units=256, activation="relu")

        self.state_values = tf.keras.layers.Dense(units=1)
        self.raw_advantages = tf.keras.layers.Dense(num_classes)
        self.advantages = tf.keras.layers.Lambda(
            lambda adv: adv - tf.reduce_max(adv, axis=1, keepdims=True),
            output_shape=lambda shape: shape
        )
        self.q_values = tf.keras.layers.Add()

        if self.is_plasticity_injected:
            self.inject_plasticity()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)

        if not self.is_plasticity_injected:
            x = self.dense2(x)
            state_values = self.state_values(x)
            raw_advantages = self.raw_advantages(x)
        else:
            x_plastic1 = self.dense2(x) + self.alpha * tf.matmul(x, self.hebb1)
            self.hebb1.assign(
                tf.matmul(tf.transpose(x), x_plastic1) * self.eta + self.hebb1 * (1 - self.eta)
            )

            state_values = self.state_values(x_plastic1) + self.alpha * tf.matmul(x_plastic1, self.hebb2_states)
            raw_advantages = self.raw_advantages(x_plastic1) + self.alpha * tf.matmul(x_plastic1, self.hebb2_adv)
            self.hebb2_states.assign(
                tf.matmul(tf.transpose(x_plastic1), state_values) * self.eta + self.hebb2_states * (1 - self.eta)
            )
            self.hebb2_adv.assign(
                tf.matmul(tf.transpose(x_plastic1), raw_advantages) * self.eta + self.hebb2_adv * (1 - self.eta)
            )

        advantages = self.advantages(raw_advantages)
        q_values = self.q_values([state_values, advantages])

        return q_values

    def inject_plasticity(self):
        self.is_plasticity_injected = True

        self.conv1.trainable = False
        self.conv2.trainable = False
        self.conv3.trainable = False
        self.flatten.trainable = False
        self.dense1.trainable = False
        self.dense2.trainable = False

        self.raw_advantages.trainable = False
        self.state_values.trainable = False

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'seed': self.seed,
            'input_shape': self.input_shape,
            'eta': self.eta,
            'alpha': self.alpha,
            'is_plasticity_injected': self.is_plasticity_injected,
            'hebb1': self.hebb1.numpy().tolist(),
            'hebb2': tf.concat([self.hebb2_adv, self.hebb2_states], axis=1).numpy().tolist()
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            num_classes=config['num_classes'],
            seed=config['seed'],
            input_shape=config['input_shape'],
            eta=config['eta'],
            alpha=config['alpha'],
            is_plasticity_injected=config['is_plasticity_injected'],
            hebb1=tf.constant(config['hebb1']) if config['hebb1'] is not None else None,
            hebb2=tf.constant(config['hebb2']) if config['hebb2'] is not None else None
        )
