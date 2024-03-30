import numpy as np

from src.data.queue import MaxSizedQueue
from src.data.replay_buffer import ReplayBuffer
from src.policy import epsilon_greedy_policy
from src.preprocess import frame_processor
from src.training import training_step


def play_one_step(env, state_queue, model, n_outputs, replay_buffer, epsilon, frame_shape):
    state_history = state_queue.get_history()
    action = epsilon_greedy_policy(state_history, model, n_outputs, epsilon)
    next_state, reward, done, truncated, info = env.step(action)

    last_state = state_history[:, :, -1][..., np.newaxis]

    preprocessed_next_state = frame_processor(next_state, shape=frame_shape)
    replay_buffer.add_experience((last_state, action, reward, preprocessed_next_state, done, truncated))
    state_queue.enqueue(preprocessed_next_state)

    return reward, done, truncated, info


def play_one_episode(episode_idx, env, model, n_steps, n_outputs, replay_buffer, history_len, frame_shape):
    obs, info = env.reset()

    state_queue = MaxSizedQueue(history_len=history_len)
    preprocessed_obs = frame_processor(obs, shape=frame_shape)
    state_queue.enqueue(preprocessed_obs)

    epsilon = max(1 - episode_idx / 500, 0.01)
    episode_rewards = []

    for step in range(n_steps):
        reward, done, truncated, info = play_one_step(env, state_queue, model, n_outputs, replay_buffer, epsilon)

        episode_rewards.append(reward)

        if done or truncated:
            break

    total_rewards = sum(episode_rewards)
    print(f"\rEpisode: {episode_idx + 1}, Steps: {n_steps + 1}, eps: {epsilon:.3f}, total rewards: {total_rewards}", end="")
    return total_rewards


def play_multiple_episodes(
        env,
        model,
        n_episodes,
        n_steps,
        n_outputs,
        history_len,
        discount_factor,
        batch_size,
        optimizer,
        loss_fn,
        replay_buffer_len,
        frame_shape):

    replay_buffer = ReplayBuffer(
        size=replay_buffer_len,
        history_len=history_len,
        batch_size=batch_size,
        sample_dim=frame_shape
    )
    rewards_over_episodes = []
    max_reward = 0
    best_weights = model.get_weights()

    for episode in range(n_episodes):
        episode_reward, furthest_step = play_one_episode(
            episode_idx=episode,
            env=env,
            model=model,
            n_steps=n_steps,
            n_outputs=n_outputs,
            replay_buffer=replay_buffer,
            history_len=history_len,
            frame_shape=frame_shape)

        rewards_over_episodes.append(episode_reward)
        if episode_reward >= max_reward:
            best_weights = model.get_weights()
            max_reward = episode_reward

        if episode > 50:
            training_step(
                model=model,
                discount_factor=discount_factor,
                optimizer=optimizer,
                loss_fn=loss_fn,
                replay_buffer=replay_buffer,
                n_outputs=n_outputs)

        model.set_weights(best_weights)

    return rewards_over_episodes
