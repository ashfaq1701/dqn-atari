from src.agent import play_multiple_episodes
from src.env import create_env
import tensorflow as tf

from src.models import get_model

HISTORY_LEN = 4
BATCH_SIZE = 32
FRAME_SHAPE = (84, 84)


def train_dqn(
        env_name,
        n_episodes=400,
        n_steps=3000,
        discount_factor=0.95,
        learning_rate=1e-4,
        model_seed=42,
        env_seed=None):
    env, _, _, action_count = create_env(env_name, env_seed)

    loss_fn = tf.keras.losses.mean_squared_error
    optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)

    model = get_model('basic_cnn', action_count, model_seed, (*FRAME_SHAPE, HISTORY_LEN))

    rewards_per_episode = play_multiple_episodes(
        env=env,
        model=model,
        n_episodes=n_episodes,
        n_steps=n_steps,
        n_outputs=action_count,
        history_len=HISTORY_LEN,
        discount_factor=discount_factor,
        batch_size=BATCH_SIZE,
        optimizer=optimizer,
        loss_fn=loss_fn,
        frame_shape=FRAME_SHAPE
    )

    return rewards_per_episode, model
