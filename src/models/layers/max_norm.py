import tensorflow as tf


class MaxNorm(tf.keras.layers.Layer):
    def call(self, x):
        return x - tf.reduce_max(x, axis=1, keepdims=True)
