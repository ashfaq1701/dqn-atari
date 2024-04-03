import numpy as np
import tensorflow as tf


class PlasticityInjectedNet(tf.keras.Model):
    def __init__(self, num_classes, eta, alpha, **kwargs):
        super(PlasticityInjectedNet, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.eta = eta
        self.alpha = alpha

        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, activation="relu")
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, activation="relu")
        self.conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation="relu")
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(units=512, activation="relu")

        self.state_values = tf.keras.layers.Dense(units=1)
        self.raw_advantages = tf.keras.layers.Dense(num_classes)
        self.advantage_max = tf.keras.layers.Lambda(lambda x: tf.reduce_max(x, axis=1, keepdims=True))
        self.advantage_norm = tf.keras.layers.Subtract()
        self.q_values = tf.keras.layers.Add()

        self.frozen_dense = None
        self.frozen_state_values = None
        self.frozen_raw_advantages = None

        self.trainable_dense = None
        self.trainable_state_values = None
        self.trainable_raw_advantages = None

        self.hebb = np.array([])

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)

        if self.frozen_dense is None:
            x = self.dense(x)
            state_values = self.state_values(x)
            raw_advantages = self.raw_advantages(x)
        else:
            x_plastic = self.trainable_dense(x)
            x = x_frozen + x_trainable - x_frozen.stop_gradient

            state_values_frozen = self.frozen_state_values(x_frozen)
            state_values_trainable = self.trainable_state_values(x_trainable)
            state_values = state_values_trainable

            raw_advantages_frozen = self.frozen_raw_advantages(x_frozen)
            raw_advantages_trainable = self.trainable_raw_advantages(x_trainable)
            raw_advantages = raw_advantages_frozen + raw_advantages_trainable - raw_advantages_frozen.stop_gradient

        advantages_max = self.advantage_max(raw_advantages)
        advantages_norm = self.advantage_centered([raw_advantages, advantages_max])
        q_values = self.q_values([state_values, advantages_norm])

        self.hebb = (1 - self.eta) * self.hebb + self.eta * tf.matmul(x, q_values, transpose_a=True)

        return q_values, self.hebb

    def inject_plasticity(self):
        self.frozen_dense = tf.keras.layers.Dense(units=512, activation="relu")
        self.frozen_dense.set_weights(self.dense.get_weights())
        self.frozen_dense.trainable = False

        self.frozen_state_values = tf.keras.layers.Dense(units=1)
        self.frozen_state_values.set_weights(self.state_values.get_weights())
        self.frozen_state_values.trainable = False

        self.frozen_raw_advantages = tf.keras.layers.Dense(self.num_classes)
        self.frozen_raw_advantages.set_weights(self.raw_advantages.get_weights())
        self.frozen_raw_advantages.trainable = False

        self.trainable_dense = tf.keras.layers.Dense(units=512, activation="relu")
        self.trainable_dense.set_weights(self.dense.get_weights())

        self.trainable_state_values = tf.keras.layers.Dense(units=1)
        self.trainable_state_values.set_weights(self.state_values.get_weights())

        self.trainable_raw_advantages = tf.keras.layers.Dense(self.num_classes)
        self.trainable_raw_advantages.set_weights(self.raw_advantages.get_weights())

        self.conv1.trainable = False
        self.conv2.trainable = False
        self.conv3.trainable = False
        self.flatten.trainable = False
        self.dense.trainable = False
        self.state_values.trainable = False
        self.raw_advantages.trainable = False


def get_model_injected_plasticity(num_classes, input_shape, eta, alpha):
    inputs = tf.keras.layers.Input(shape=input_shape)
    hebb = tf.keras.layers.Input(shape=(512, num_classes))

    model = PlasticityInjectedNet(num_classes, eta, alpha)
    q_values, updated_hebb = model(inputs, hebb)

    return tf.keras.Model(inputs=[inputs, hebb], outputs=[q_values, updated_hebb])