import numpy as np

from src.policy import epsilon_greedy_policy


def play_one_step(env, state_queue, model, n_outputs, replay_buffer, epsilon):
    state_history = state_queue.get_history()
    action = epsilon_greedy_policy(state_history, model, n_outputs, epsilon)
    next_state, reward, done, truncated, info = env.step(action)

    last_state = state_history[:, :, -1][..., np.newaxis]
    replay_buffer.add_experience((last_state, action, reward, next_state, done, truncated))
    state_queue.enqueue(next_state)
    return reward, done, truncated, info
