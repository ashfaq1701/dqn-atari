import numpy as np

from src.agents.dqn_agent import play_multiple_episodes_dqn
from src.data.queue import MaxSizedQueue
from src.policy import epsilon_greedy_policy
from src.preprocess import frame_processor
from src.training import training_step


def play_multiple_episodes_dqn_plastic(
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
        frame_shape,
        initial_training_percentage,
        replay_buff_max_len
):
    rewards_over_episodes, steps_over_episodes, avg_max_q_values = play_multiple_episodes_dqn(
        env,
        model,
        target_model,
        int(n_episodes * initial_training_percentage),
        n_steps,
        n_outputs,
        history_len,
        discount_factor,
        batch_size,
        optimizer,
        loss_fn,
        frame_shape,
        replay_buff_max_len
    )

    model.inject_plasticity()

    rewards_over_episodes, steps_over_episodes, avg_max_q_values = play_multiple_episodes_dqn_inject_plasticity(
        env,
        model,
        target_model,
        int(n_episodes * (1.0 - initial_training_percentage)),
        int(n_episodes * initial_training_percentage),
        n_steps,
        n_outputs,
        history_len,
        discount_factor,
        batch_size,
        optimizer,
        loss_fn,
        frame_shape,
        rewards_over_episodes,
        steps_over_episodes,
        avg_max_q_values
    )

    return rewards_over_episodes, steps_over_episodes, avg_max_q_values


def play_multiple_episodes_dqn_inject_plasticity(
        env,
        model,
        target_model,
        n_episodes,
        start_episode,
        n_steps,
        n_outputs,
        history_len,
        discount_factor,
        batch_size,
        optimizer,
        loss_fn,
        frame_shape,
        rewards_over_episodes_pre_training,
        steps_over_episodes_pre_training,
        avg_max_q_value_pre_training
):
    rewards_over_episodes = rewards_over_episodes_pre_training
    steps_over_episodes = steps_over_episodes_pre_training
    avg_max_q_values = avg_max_q_value_pre_training

    for episode in range(start_episode, start_episode + n_episodes):
        experiences, episode_reward, max_step_of_episode = play_one_episode(
            episode_idx=episode,
            env=env,
            model=model,
            n_steps=n_steps,
            n_outputs=n_outputs,
            history_len=history_len,
            frame_shape=frame_shape
        )

        rewards_over_episodes.append(episode_reward)
        steps_over_episodes.append(max_step_of_episode)

        windowed_experiences = shuffle_and_batch_experiences(experiences, batch_size)

        total_avg_max_q_value = 0
        for experience_window in windowed_experiences:
            unpacked_experience_window = [np.array(item) for item in zip(*experience_window)]
            total_avg_max_q_value += training_step(
                model=model,
                target_model=target_model,
                experiences=unpacked_experience_window,
                discount_factor=discount_factor,
                optimizer=optimizer,
                loss_fn=loss_fn,
                n_outputs=n_outputs
            )

        avg_max_q_values.append(total_avg_max_q_value / len(windowed_experiences))

    return rewards_over_episodes, rewards_over_episodes, avg_max_q_values


def play_one_step(env, state_queue, model, n_outputs, frame_shape):
    state_history = state_queue.get_history()
    action = epsilon_greedy_policy(state_history, model, n_outputs, 0.0)
    next_state, reward, done, truncated, info = env.step(action)
    preprocessed_next_state = frame_processor(next_state, shape=frame_shape)
    state_queue.enqueue(preprocessed_next_state)
    next_state_history = state_queue.get_history()
    return state_history, action, reward, done, next_state_history, truncated


def play_one_episode(episode_idx, env, model, n_steps, n_outputs, history_len, frame_shape):
    obs, info = env.reset()

    state_queue = MaxSizedQueue(history_len=history_len)
    preprocessed_obs = frame_processor(obs, shape=frame_shape)
    state_queue.enqueue(preprocessed_obs)

    episode_rewards = []
    experiences = []
    max_step = 0

    for step in range(n_steps):
        state_history, action, reward, done, next_state_history, truncated = play_one_step(
            env,
            state_queue,
            model,
            n_outputs,
            frame_shape
        )

        max_step = step
        episode_rewards.append(reward)

        experiences.append((
            state_history,
            action,
            reward,
            next_state_history,
            done,
            truncated
        ))

        if done or truncated:
            break

    total_rewards = sum(episode_rewards)
    print(f"\rEpisode: {episode_idx + 1}, Steps: {max_step + 1}, eps: 1.000, total rewards: {total_rewards}")
    return experiences, total_rewards, max_step


def shuffle_and_batch_experiences(experiences, batch_size):
    indices = list(range(len(experiences)))
    indices_np = np.array(indices)
    np.random.shuffle(indices_np)

    windows = []
    for i in range(0, len(indices_np), batch_size):
        window = []
        for j in range(i, min(i + batch_size, len(indices_np))):
            window.append(experiences[indices_np[j]])
        windows.append(window)

    return windows
