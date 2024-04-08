import tensorflow as tf
from functools import partial
from keras.src.layers import Rescaling


def get_model_dqn(num_classes, seed, input_shape):
    tf.random.set_seed(seed)

    DefaultConv2D = partial(tf.keras.layers.Conv2D, kernel_size=3, padding="same", activation="relu")

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        Rescaling(1. / 255),

        DefaultConv2D(filters=32, kernel_size=7),
        tf.keras.layers.MaxPool2D(),
        DefaultConv2D(filters=64),
        DefaultConv2D(filters=64),
        tf.keras.layers.MaxPool2D(),
        DefaultConv2D(filters=128),
        DefaultConv2D(filters=128),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=64, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(units=32, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(units=num_classes)
    ])

    return model
