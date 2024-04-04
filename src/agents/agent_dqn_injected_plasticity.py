from src.agents.dqn_agent import play_multiple_episodes_dqn


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
        frame_shape
):
    model.inject_plasticity()
    rewards_over_episodes, steps_over_episodes, avg_max_q_values = play_multiple_episodes_dqn(
        env,
        model,
        target_model,
        n_episodes // 2,
        n_steps,
        n_outputs,
        history_len,
        discount_factor,
        batch_size,
        optimizer,
        loss_fn,
        frame_shape
    )

    rewards_over_episodes, steps_over_episodes, avg_max_q_values = play_multiple_episodes_dqn_inject_plasticity(
        env,
        model,
        target_model,
        n_episodes // 2,
        n_episodes // 2,
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
    return rewards_over_episodes_pre_training, steps_over_episodes_pre_training, avg_max_q_value_pre_training


def play_one_step():
    pass


def play_one_episode():
    pass
