import keras
from src.agent import play_multiple_episodes
from src.env import create_env
import tensorflow as tf

from src.models import get_model

HISTORY_LEN = 4
BATCH_SIZE = 32
FRAME_SHAPE = (84, 84)


def train_dqn(
        env_name,
        n_episodes=2000,
        n_steps=3000,
        discount_factor=0.95,
        learning_rate=1e-4,
        model_seed=42,
        env_seed=None):
    env, _, _, action_count = create_env(env_name, env_seed)

    loss_fn = tf.keras.losses.mean_squared_error
    optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)

    keras.config.enable_unsafe_deserialization()
    model = get_model('dueling_dqn', action_count, model_seed, (*FRAME_SHAPE, HISTORY_LEN))
    target_model = tf.keras.models.clone_model(model)
    target_model.set_weights(model.get_weights())

    rewards_per_episode, steps_over_episode = play_multiple_episodes(
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
        frame_shape=FRAME_SHAPE
    )

    return rewards_per_episode, steps_over_episode, model
