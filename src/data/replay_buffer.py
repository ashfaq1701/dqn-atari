import numpy as np


def find_first_greater_index(arr, value):
    for i, element in enumerate(arr):
        if element > value:
            return i
    return -1


class ReplayBuffer:
    def __init__(self, history_len=4, batch_size=32, sample_dim=(84, 84)):
        self.history_len = history_len
        self.batch_size = batch_size
        self.sample_dim = sample_dim

        self.episode_runs = []

        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.truncateds = []

    def add_experience(self, experience):
        state, action, reward, next_state, done, truncated = experience

        if self._is_episode_start():
            self.states.append(state)

        self.states.append(next_state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.truncateds.append(truncated)

    def _get_experience(self, idx):
        sample_idx = self._get_sample_idx(idx)
        left_bound, right_bound = self._get_episode_sample_boundaries(idx)
        state_history = self._get_slice(sample_idx - self.history_len, sample_idx - 1, left_bound, right_bound)
        next_state_history = self._get_slice(sample_idx - self.history_len + 1, sample_idx, left_bound, right_bound)
        return (
            state_history,
            self.actions[idx],
            self.rewards[idx],
            next_state_history,
            self.dones[idx],
            self.truncateds[idx]
        )

    def sample_experiences(self):
        indices = np.random.choice(range(len(self.actions)), size=self.batch_size)
        experiences = [self._get_experience(idx) for idx in indices]
        return zip(*experiences)

    def end_episode(self):
        self.episode_runs.append(len(self.actions))

    def _get_count_till_last_episode(self):
        if len(self.episode_runs) == 0:
            return 0

        return self.episode_runs[-1]

    def _is_episode_start(self):
        return len(self.actions) == self._get_count_till_last_episode()

    def _get_sample_idx(self, idx):
        episode_idx = find_first_greater_index(self.episode_runs, idx)
        return idx + episode_idx + 1

    def _get_episode_sample_boundaries(self, idx):
        episode_idx = find_first_greater_index(self.episode_runs, idx)
        left_bound = (self.episode_runs[episode_idx - 1] if episode_idx > 0 else 0) + episode_idx
        right_bound = self.episode_runs[episode_idx] + episode_idx + 1
        return left_bound, right_bound

    def _get_slice(self, start, end, left_bound, right_bound):
        state_slice = []

        for current in range(start, end + 1):
            if current < left_bound or current < 0:
                state_slice.append(self.states[left_bound])
            elif current > right_bound or current > len(self.states) - 1:
                state_slice.append(self.states[right_bound])
            else:
                state_slice.append(self.states[current])

        return np.concatenate(state_slice, axis=-1)
