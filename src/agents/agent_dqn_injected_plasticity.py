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
        replay_buff_max_len,
        plasticity_training_epsilon,
        restore_best_weights
):
    rewards_over_episodes, steps_over_episodes, avg_max_q_values, losses = play_multiple_episodes_dqn(
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
        replay_buff_max_len,
        restore_best_weights
    )

    model.inject_plasticity()

    rewards_over_episodes, steps_over_episodes, avg_max_q_values, losses = play_multiple_episodes_dqn_inject_plasticity(
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
        plasticity_training_epsilon,
        optimizer,
        loss_fn,
        frame_shape,
        rewards_over_episodes,
        steps_over_episodes,
        avg_max_q_values,
        losses
    )

    return rewards_over_episodes, steps_over_episodes, avg_max_q_values, losses


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
        plasticity_training_epsilon,
        optimizer,
        loss_fn,
        frame_shape,
        rewards_over_episodes_pre_training,
        steps_over_episodes_pre_training,
        avg_max_q_value_pre_training,
        losses_pre_training
):
    rewards_over_episodes = rewards_over_episodes_pre_training[:]
    steps_over_episodes = steps_over_episodes_pre_training[:]
    avg_max_q_values = avg_max_q_value_pre_training[:]
    losses = losses_pre_training[:]

    for episode in range(start_episode, start_episode + n_episodes):
        experiences, episode_reward, max_step_of_episode = play_one_episode(
            episode_idx=episode,
            env=env,
            model=model,
            n_steps=n_steps,
            n_outputs=n_outputs,
            history_len=history_len,
            frame_shape=frame_shape,
            plasticity_training_epsilon=plasticity_training_epsilon
        )

        rewards_over_episodes.append(episode_reward)
        steps_over_episodes.append(max_step_of_episode)

        sampled_experiences = sample_experiences(experiences, batch_size)

        loss, avg_max_q_value = training_step(
            model=model,
            target_model=target_model,
            experiences=sampled_experiences,
            discount_factor=discount_factor,
            optimizer=optimizer,
            loss_fn=loss_fn,
            n_outputs=n_outputs
        )

        if episode % 50 == 0:
            target_model.set_weights(model.get_weights())

        avg_max_q_values.append(avg_max_q_value)
        losses.append(loss)

    return rewards_over_episodes, rewards_over_episodes, avg_max_q_values, losses


def play_one_step(env, state_queue, model, n_outputs, frame_shape, plasticity_training_epsilon):
    state_history = state_queue.get_history()
    action = epsilon_greedy_policy(state_history, model, n_outputs, plasticity_training_epsilon)
    next_state, reward, done, truncated, info = env.step(action)
    preprocessed_next_state = frame_processor(next_state, shape=frame_shape)
    state_queue.enqueue(preprocessed_next_state)
    next_state_history = state_queue.get_history()
    return state_history, action, reward, done, next_state_history, truncated


def play_one_episode(
        episode_idx,
        env,
        model,
        n_steps,
        n_outputs,
        history_len,
        frame_shape,
        plasticity_training_epsilon
):
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
            frame_shape,
            plasticity_training_epsilon
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
    print(f"\rEpisode: {episode_idx + 1}, Steps: {max_step + 1}, eps: {plasticity_training_epsilon:.3f}, total rewards: {total_rewards}")
    return experiences, total_rewards, max_step


def sample_experiences(experiences, batch_size):
    indices = np.random.choice(range(len(experiences)), size=batch_size)
    sampled_experiences = [experiences[idx] for idx in indices]
    return [np.array(item) for item in zip(*sampled_experiences)]
