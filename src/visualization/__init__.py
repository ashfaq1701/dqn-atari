import matplotlib.pyplot as plt


def plot_rewards(rewards, title='Rewards', xlabel='Timestep', ylabel='Reward'):
    timesteps = list(range(1, len(rewards) + 1))

    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    plt.plot(timesteps, rewards, label='Rewards')

    # Add labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # Add legend
    plt.legend()
    plt.show()


def plot_q(q_values, title='Avg Max Q Values', xlabel='Timestep', ylabel='Q Value'):
    timesteps = list(range(51, len(q_values) + 1))

    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    plt.plot(timesteps, q_values[50:], label='Rewards')

    # Add labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # Add legend
    plt.legend()
    plt.show()


def plot_all_rewards(spaceinvaders_rewards, breakout_rewards, pong_rewards, seaquest_rewards):
    n_episodes = len(seaquest_rewards)
    x = range(1, n_episodes + 1)

    # Assuming seaquest_rewards, pong_rewards, breakout_rewards, and spaceinvaders_rewards are defined

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))  # Create a 2x2 grid plot

    # Plot SpaceInvaders Rewards
    axs[0, 0].plot(x, spaceinvaders_rewards)
    axs[0, 0].set_title('SpaceInvaders Rewards')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Total Rewards')

    # Plot Breakout Rewards
    axs[0, 1].plot(x, breakout_rewards)
    axs[0, 1].set_title('Breakout Rewards')
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('Total Rewards')

    # Plot Pong Rewards
    axs[1, 0].plot(x, pong_rewards)
    axs[1, 0].set_title('Pong Rewards')
    axs[1, 0].set_xlabel('Episode')
    axs[1, 0].set_ylabel('Total Rewards')

    # Plot SeaQuest Rewards
    axs[1, 1].plot(x, seaquest_rewards)
    axs[1, 1].set_title('SeaQuest Rewards')
    axs[1, 1].set_xlabel('Episode')
    axs[1, 1].set_ylabel('Total Rewards')

    plt.tight_layout()
    plt.show()
