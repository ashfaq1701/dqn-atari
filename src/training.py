import numpy as np
import tensorflow as tf


def training_step(model, discount_factor, optimizer, loss_fn, replay_buffer, n_outputs):
    experiences = replay_buffer.sample_experiences()
    states, actions, rewards, next_states, dones, truncateds = experiences

    next_states = [np.expand_dims(state, axis=0) for state in next_states]
    next_Q_values = model.predict(next_states, verbose=0)

    max_next_Q_values = next_Q_values.max(axis=1)
    runs = 1.0 - (dones | truncateds)
    target_Q_values = rewards + runs * discount_factor * max_next_Q_values
    target_Q_values = target_Q_values.reshape(-1, 1)
    mask = tf.one_hot(actions, n_outputs)

    with tf.GradientTape() as tape:
        all_Q_values = model(states)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
