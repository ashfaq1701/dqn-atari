import numpy as np

from src.data.queue import MaxSizedQueue
from src.data.replay_buffer import ReplayBuffer
from src.policy import epsilon_greedy_policy
from src.preprocess import frame_processor
from src.training import training_step


def play_multiple_episodes_dqn(
        env,
        model,
        target_model,
        n_episodes,
        n_steps,
        n_outputs,
        history_len,
        discount_factor,
        batch_size,
        optimizer,
        loss_fn,
        frame_shape):
    replay_buffer = ReplayBuffer(
        history_len=history_len,
        batch_size=batch_size,
        sample_dim=frame_shape
    )
    rewards_over_episodes = []
    steps_over_episodes = []
    max_reward = float('-inf')
    best_weights = model.get_weights()
    avg_max_q_values = []

    for episode in range(n_episodes):
        epsilon = max(1 - episode / int(0.8 * n_episodes), 0.01)

        episode_reward, max_step_of_episode = play_one_episode(
            episode_idx=episode,
            env=env,
            model=model,
            n_steps=n_steps,
            n_outputs=n_outputs,
            epsilon=epsilon,
            replay_buffer=replay_buffer,
            history_len=history_len,
            frame_shape=frame_shape
        )
        replay_buffer.end_episode()

        rewards_over_episodes.append(episode_reward)
        steps_over_episodes.append(max_step_of_episode)

        if episode_reward >= max_reward:
            best_weights = model.get_weights()
            max_reward = episode_reward

        avg_max_q_value = 0
        if episode >= 50:
            experiences = replay_buffer.sample_experiences()
            avg_max_q_value = training_step(
                model=model,
                target_model=target_model,
                experiences=experiences,
                discount_factor=discount_factor,
                optimizer=optimizer,
                loss_fn=loss_fn,
                n_outputs=n_outputs
            )

            if episode % 50 == 0:
                target_model.set_weights(model.get_weights())

        avg_max_q_values.append(avg_max_q_value)

    model.set_weights(best_weights)
    return rewards_over_episodes, steps_over_episodes, avg_max_q_values


def play_one_step(env, state_queue, model, n_outputs, replay_buffer, epsilon, frame_shape):
    state_history = state_queue.get_history()
    action = epsilon_greedy_policy(state_history, model, n_outputs, epsilon)
    next_state, reward, done, truncated, info = env.step(action)

    last_state = state_history[:, :, -1][..., np.newaxis]

    preprocessed_next_state = frame_processor(next_state, shape=frame_shape)
    replay_buffer.add_experience((last_state, action, reward, preprocessed_next_state, done, truncated))
    state_queue.enqueue(preprocessed_next_state)

    return reward, done, truncated, info


def play_one_episode(episode_idx, env, model, n_steps, n_outputs, epsilon, replay_buffer, history_len, frame_shape):
    obs, info = env.reset()

    state_queue = MaxSizedQueue(history_len=history_len)
    preprocessed_obs = frame_processor(obs, shape=frame_shape)
    state_queue.enqueue(preprocessed_obs)

    episode_rewards = []
    max_step = 0

    for step in range(n_steps):
        reward, done, truncated, info = play_one_step(
            env,
            state_queue,
            model,
            n_outputs,
            replay_buffer,
            epsilon,
            frame_shape
        )

        max_step = step
        episode_rewards.append(reward)

        if done or truncated:
            break

    total_rewards = sum(episode_rewards)
    print(f"\rEpisode: {episode_idx + 1}, Steps: {max_step + 1}, eps: {epsilon:.3f}, total rewards: {total_rewards}")
    return total_rewards, max_step
