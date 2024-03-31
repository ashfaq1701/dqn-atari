import tensorflow as tf


def get_model_duelling_dqn(num_classes, seed, input_shape):
    tf.random.set_seed(seed)

    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, activation="relu")(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation="relu")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=512, activation="relu")(x)

    state_values = tf.keras.layers.Dense(units=1)(x)
    raw_advantages = tf.keras.layers.Dense(num_classes)(x)

    advantages = tf.keras.layers.Lambda(
        lambda adv: adv - tf.reduce_max(adv, axis=1, keepdims=True),
        output_shape=lambda shape: shape
    )(raw_advantages)

    Q_values = tf.keras.layers.Add()([state_values, advantages])

    model = tf.keras.Model(inputs=[inputs], outputs=[Q_values])

    return model
