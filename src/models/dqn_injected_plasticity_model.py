import numpy as np
import tensorflow as tf
from src.models.layers.max_norm import MaxNorm


@tf.keras.utils.register_keras_serializable(package="src.models")
class DDQNInjectedPlasticityModel(tf.keras.Model):
    def __init__(
            self,
            num_classes,
            seed,
            input_shape,
            eta,
            alpha,
            is_plasticity_injected=False,
            hebb_dense1=None,
            hebb_state_values=None,
            hebb_raw_advantages=None,
            **kwargs
    ):
        super(DDQNInjectedPlasticityModel, self).__init__(**kwargs)

        self.seed = seed
        if seed is not None:
            tf.random.set_seed(seed)

        self.num_classes = num_classes
        self.input_shape = input_shape
        self.eta = eta
        self.alpha = alpha
        self.is_plasticity_injected = is_plasticity_injected

        if hebb_dense1 is None:
            self.hebb_dense1 = self.add_weight(
                shape=(3136, 512),
                initializer=tf.zeros_initializer(),
                trainable=True,
                name="hebb_dense1"
            )
        else:
            self.hebb_dense1 = self.add_weight(
                shape=(3136, 512),
                initializer=tf.constant_initializer(hebb_dense1),
                trainable=True,
                name="hebb_dense1"
            )

        if hebb_state_values is None:
            self.hebb_state_values = self.add_weight(
                shape=(512, 1),
                initializer=tf.zeros_initializer(),
                trainable=True,
                name="hebb_state_values"
            )
        else:
            self.hebb_state_values = self.add_weight(
                shape=(512, 1),
                initializer=tf.constant_initializer(hebb_state_values),
                trainable=True,
                name="hebb_dense1"
            )

        if hebb_raw_advantages is None:
            self.hebb_raw_advantages = self.add_weight(
                shape=(512, num_classes),
                initializer=tf.zeros_initializer(),
                trainable=True,
                name="hebb_raw_advantages"
            )
        else:
            self.hebb_raw_advantages = self.add_weight(
                shape=(512, num_classes),
                initializer=tf.constant_initializer(hebb_raw_advantages),
                trainable=True,
                name="hebb_dense1"
            )

        self.rescaling = tf.keras.layers.Rescaling(1. / 255)
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
        self.dense1 = tf.keras.layers.Dense(units=512)

        self.state_values = tf.keras.layers.Dense(units=1)
        self.raw_advantages = tf.keras.layers.Dense(num_classes)
        self.advantages = MaxNorm()
        self.q_values = tf.keras.layers.Add()

        if self.is_plasticity_injected:
            self.inject_plasticity()

    def call(self, inputs):
        x = self.rescaling(inputs)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)

        if not self.is_plasticity_injected:
            dense1_op = tf.nn.relu(self.dense1(x))
            state_values_op = self.state_values(dense1_op)
            raw_advantages_op = self.raw_advantages(dense1_op)
        else:
            dense1_op = tf.nn.relu(
                self.dense1(x) + self.alpha * tf.matmul(x, self.hebb_dense1)
            )
            self.hebb_dense1.assign(
                tf.matmul(tf.transpose(x), dense1_op) * self.eta + self.hebb_dense1 * (1 - self.eta)
            )

            state_values_op = (self.state_values(dense1_op)
                               + self.alpha * tf.matmul(dense1_op, self.hebb_state_values))
            self.hebb_state_values.assign(
                tf.matmul(tf.transpose(dense1_op), state_values_op) * self.eta + self.hebb_state_values * (1 - self.eta)
            )

            raw_advantages_op = (self.raw_advantages(dense1_op)
                                 + self.alpha * tf.matmul(dense1_op, self.hebb_raw_advantages))
            self.hebb_raw_advantages.assign(
                tf.matmul(tf.transpose(state_values_op), raw_advantages_op) * self.eta
                + self.hebb_raw_advantages * (1 - self.eta)
            )

        advantages = self.advantages(raw_advantages_op)
        q_values = self.q_values([state_values_op, advantages])

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

        self.advantages.trainable = False
        self.q_values.trainable = False

        self._reinitialize_hebb_tensors()

        self.hebb_dense1.trainable = True
        self.hebb_state_values.trainable = True
        self.hebb_raw_advantages.trainable = True

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'seed': self.seed,
            'input_shape': self.input_shape,
            'eta': self.eta,
            'alpha': self.alpha,
            'is_plasticity_injected': self.is_plasticity_injected,
            'hebb_dense1': self.hebb_dense1.numpy().tolist(),
            'hebb_state_values': self.hebb_state_values.numpy().tolist(),
            'hebb_raw_advantages': self.hebb_raw_advantages.numpy().tolist()
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
            hebb_dense1=np.array(config['hebb_dense1'])
            if config['hebb_dense1'] is not None else None,
            hebb_state_values=np.array(config['hebb_state_values'])
            if config['hebb_state_values'] is not None else None,
            hebb_raw_advantages=np.array(config['hebb_raw_advantages'])
            if config['hebb_raw_advantages'] is not None else None
        )

    def _reinitialize_hebb_tensors(self):
        self.hebb_dense1.assign(tf.zeros(shape=self.hebb_dense1.shape))
        self.hebb_state_values.assign(tf.zeros(shape=self.hebb_state_values.shape))
        self.hebb_raw_advantages.assign(tf.zeros(shape=self.hebb_raw_advantages.shape))