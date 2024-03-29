import gymnasium as gym


def create_env(env_name, env_seed=None):
    env = gym.make(env_name, render_mode="rgb_array")
    action_space = env.action_space
    action_count = action_space.n if isinstance(action_space, gym.spaces.Discrete) else -1

    if env_seed is not None:
        obs, info = env.reset(seed=env_seed)
    else:
        obs, info = env.reset()

    return env, obs, info, action_count