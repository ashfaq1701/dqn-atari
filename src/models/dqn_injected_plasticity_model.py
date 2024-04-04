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
            hebb=None,
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

        if hebb is None:
            self.hebb = tf.Variable(tf.zeros(shape=(512, num_classes + 1)), trainable=False)
        else:
            self.hebb = hebb

        self.hebb_adv = tf.Variable(self.hebb[:, :num_classes], trainable=False)
        self.hebb_states = tf.Variable(self.hebb[:, num_classes:], trainable=False)

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
            state_values = self.state_values(x)
            raw_advantages = self.raw_advantages(x)
        else:
            state_values = self.state_values(x) + self.alpha * tf.matmul(x, self.hebb_states)
            raw_advantages = self.raw_advantages(x) + self.alpha * tf.matmul(x, self.hebb_adv)

            self.hebb_states.assign(
                tf.matmul(tf.transpose(x), state_values) * self.eta + self.hebb_states * (1 - self.eta)
            )
            self.hebb_adv.assign(
                tf.matmul(tf.transpose(x), raw_advantages) * self.eta + self.hebb_adv * (1 - self.eta)
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
            'hebb': tf.concat([self.hebb_adv, self.hebb_states], axis=1).numpy().tolist()
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
            hebb=tf.constant(config['hebb']) if config['hebb'] is not None else None
        )
