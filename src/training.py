import numpy as np
import tensorflow as tf


def training_step(model, target_model, discount_factor, optimizer, loss_fn, replay_buffer, n_outputs):
    experiences = replay_buffer.sample_experiences()
    states, actions, rewards, next_states, dones, truncateds = experiences

    next_Q_values = model.predict(next_states, verbose=0)
    best_next_actions = next_Q_values.argmax(axis=1)
    next_mask = tf.one_hot(best_next_actions, n_outputs).numpy()
    max_next_Q_values = (target_model.predict(next_states, verbose=0) * next_mask).sum(axis=1)

    runs = 1.0 - (dones | truncateds)
    target_Q_values = rewards + runs * discount_factor * max_next_Q_values
    target_Q_values = target_Q_values.reshape(-1, 1)
    mask = tf.one_hot(actions, n_outputs)

    with tf.GradientTape() as tape:
        all_Q_values = model(states)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        avg_max_q_value = np.mean(tf.reduce_max(Q_values, axis=1).numpy())
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return avg_max_q_value
