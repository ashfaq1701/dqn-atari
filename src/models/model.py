import tensorflow as tf


def get_model_basic_cnn(num_classes, seed, input_shape):
    tf.random.set_seed(seed)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(filters=16, kernel_size=8, strides=4, activation="relu"),
        tf.keras.layers.Conv2D(filters=32, kernel_size=4, strides=2, activation="relu"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=256, activation="relu"),
        tf.keras.layers.Dense(units=num_classes, activation="softmax")
    ])

    return model
