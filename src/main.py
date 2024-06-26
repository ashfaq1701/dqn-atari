import keras
import tensorflow as tf

from src.agents.agent_dqn_injected_plasticity import play_multiple_episodes_dqn_plastic
from src.agents.dqn_agent import play_multiple_episodes_dqn
from src.env import create_env
from src.models import get_model

HISTORY_LEN = 4
BATCH_SIZE = 32
FRAME_SHAPE = (84, 84)


def train_dqn(
        env_name,
        method,
        n_episodes=10_000,
        n_steps=3000,
        discount_factor=0.95,
        initial_learning_rate=0.01,
        final_learning_rate=1e-4,
        model_seed=42,
        env_seed=None,
        replay_buff_max_len=50_000,
        initial_training_percentage=0.7,
        eta=1e-3,
        alpha=0.2,
        plasticity_training_epsilon=0.0,
        restore_best_weights=True,
        loss_fn=tf.keras.losses.mean_squared_error):

    env, _, _, action_count = create_env(env_name, env_seed)

    decay_rate = (final_learning_rate / initial_learning_rate) ** (1 / (0.6 * n_episodes))
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=int(0.6 * n_episodes),
        decay_rate=decay_rate,
        staircase=False
    )

    optimizer = tf.keras.optimizers.Nadam(learning_rate=lr_schedule)
    keras.config.enable_unsafe_deserialization()

    if method == 'dqn':
        model = get_model(
            'dqn',
            action_count,
            model_seed,
            (*FRAME_SHAPE, HISTORY_LEN),
            None,
            None
        )
    elif method == 'ddqn':
        model = get_model(
            'ddqn',
            action_count,
            model_seed,
            (*FRAME_SHAPE, HISTORY_LEN),
            None,
            None
        )
    elif method == 'ddqn_injected_plasticity':
        model = get_model(
            'ddqn_injected_plasticity',
            action_count,
            model_seed,
            (*FRAME_SHAPE, HISTORY_LEN),
            eta,
            alpha
        )
    else:
        raise Exception("Unknown method: {}".format(method))

    target_model = tf.keras.models.clone_model(model)
    target_model.set_weights(model.get_weights())

    if method == 'dqn' or method == 'ddqn':
        rewards_per_episode, steps_over_episode, q_values_over_episode, losses = play_multiple_episodes_dqn(
            env=env,
            model=model,
            target_model=model,
            n_episodes=n_episodes,
            n_steps=n_steps,
            n_outputs=action_count,
            history_len=HISTORY_LEN,
            discount_factor=discount_factor,
            batch_size=BATCH_SIZE,
            optimizer=optimizer,
            loss_fn=loss_fn,
            frame_shape=FRAME_SHAPE,
            replay_buff_max_len=replay_buff_max_len,
            restore_best_weights=restore_best_weights
        )
    elif method == 'ddqn_injected_plasticity':
        rewards_per_episode, steps_over_episode, q_values_over_episode, losses = play_multiple_episodes_dqn_plastic(
            env=env,
            model=model,
            target_model=model,
            n_episodes=n_episodes,
            n_steps=n_steps,
            n_outputs=action_count,
            history_len=HISTORY_LEN,
            discount_factor=discount_factor,
            batch_size=BATCH_SIZE,
            optimizer=optimizer,
            loss_fn=loss_fn,
            frame_shape=FRAME_SHAPE,
            initial_training_percentage=initial_training_percentage,
            replay_buff_max_len=replay_buff_max_len,
            plasticity_training_epsilon=plasticity_training_epsilon,
            restore_best_weights=restore_best_weights
        )
    else:
        raise Exception("Unknown method: {}".format(method))

    return rewards_per_episode, steps_over_episode, q_values_over_episode, losses, model
