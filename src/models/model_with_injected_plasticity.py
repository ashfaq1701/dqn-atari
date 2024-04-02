import tensorflow as tf


class PlasticityInjectedNet(tf.keras.Model):
    def __init__(self, encoder, head, eta, alpha):
        super(PlasticityInjectedNet, self).__init__()
        self.encoder = encoder
        self.head = head
        self.head_frozen = None
        self.head_trainable = None
        self.eta = eta
        self.alpha = alpha

    def call(self, x, hebb):
        features = self.encoder(x)
        if self.head_frozen is None:
            return self.head(features), hebb
        else:
            hactiv = self.head(features) + self.head_trainable(features) - self.head_frozen(features)
            hebb = (1 - self.eta) * hebb + self.eta * tf.matmul(features, hactiv, transpose_a=True)
            return hactiv, hebb

    def inject_plasticity(self):
        self.head_frozen = tf.keras.models.clone_model(self.head)
        self.head_frozen.trainable = False

        self.head_trainable = tf.keras.models.clone_model(self.head)

        self.encoder.trainable = False
        self.head.trainable = False


def get_model_injected_plasticity(num_classes, seed, input_shape, eta, alpha):
    tf.random.set_seed(seed)

    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, activation="relu")(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation="relu")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=512, activation="relu")(x)

    encoder = tf.keras.Model(inputs=[inputs], outputs=[x])

    state_values = tf.keras.layers.Dense(units=1)(x)
    raw_advantages = tf.keras.layers.Dense(num_classes)(x)

    advantages = tf.keras.layers.Lambda(
        lambda adv: adv - tf.reduce_max(adv, axis=1, keepdims=True),
        output_shape=lambda shape: shape
    )(raw_advantages)

    Q_values = tf.keras.layers.Add()([state_values, advantages])

    head = tf.keras.Model(inputs=[x], outputs=[Q_values])

    model = PlasticityInjectedNet(encoder, head, eta, alpha)

    return model
