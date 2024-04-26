import matplotlib
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from src.data.queue import MaxSizedQueue
from src.env import create_env
from src.preprocess import frame_processor


def simulate_playing_game(model_path, env_name, env_seed=None, history_len=4, max_steps=10_000, frame_shape=(84, 84)):
    model = tf.keras.models.load_model(model_path, custom_objects={'tf': tf})

    env, _, _, action_count = create_env(env_name, env_seed)
    obs, info = env.reset()

    frames = [obs]

    state_queue = MaxSizedQueue(history_len=history_len)
    preprocessed_obs = frame_processor(obs, shape=frame_shape)

    state_queue.enqueue(preprocessed_obs)

    total_rewards = 0

    for step in range(max_steps):
        state_history = state_queue.get_history()
        Q_values = model.predict(state_history[np.newaxis], verbose=0)[0]
        action = Q_values.argmax()

        next_state, reward, done, truncated, info = env.step(action)
        preprocessed_next_state = frame_processor(next_state, shape=frame_shape)
        state_queue.enqueue(preprocessed_next_state)

        total_rewards += reward

        frames.append(next_state)
        if step == 250:
            plt.imsave('../data/outputs/game_frame_result.png', next_state)

        if done or truncated:
            break

    anim = plot_animation(frames)
    return anim, total_rewards


def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,


def plot_animation(frames, repeat=False, interval=40):
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    anim = FuncAnimation(
        fig,
        update_scene,
        fargs=(frames, patch),
        frames=len(frames),
        repeat=repeat,
        interval=interval
    )
    plt.close()
    return anim
