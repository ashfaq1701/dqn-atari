import numpy as np
from collections import deque


class ReplayBuffer:
    def __init__(self, history_len=4, batch_size=32, maxlen=100_000):
        self.history_len = history_len
        self.batch_size = batch_size
        self.maxlen = maxlen

        self.experiences = deque(maxlen=maxlen)

    def add_experience(self, experience, is_start_of_episode):
        state, action, reward, next_state, done, truncated = experience
        state_sequence = self._get_state_sequence(state, is_start_of_episode)
        next_state_sequence = self._get_next_sample_sequence(next_state, state, is_start_of_episode)
        self.experiences.append((state_sequence, action, reward, next_state_sequence, done, truncated))

    def _get_experience(self, idx):
        states, action, reward, next_states, done, truncated = self.experiences[idx]

        return (
            np.concatenate(states, axis=-1),
            action,
            reward,
            np.concatenate(next_states, axis=-1),
            done,
            truncated
        )

    def _get_state_sequence(self, state, is_start_of_episode):
        if is_start_of_episode or len(self.experiences) == 0:
            return [np.copy(state) for _ in range(self.history_len)]

        last_sequence = self.experiences[-1][0]
        return [np.copy(last_sequence[i]) for i in range(1, self.history_len)] + [state]

    def _get_next_sample_sequence(self, next_state, state, is_start_of_episode):
        if is_start_of_episode or len(self.experiences) == 0:
            return [np.copy(state) for _ in range(self.history_len - 1)] + [next_state]

        last_sequence = self.experiences[-1][3]
        return [np.copy(last_sequence[i]) for i in range(1, self.history_len)] + [next_state]

    def sample_experiences(self):
        indices = np.random.choice(range(len(self.experiences)), size=self.batch_size)
        experiences = [self._get_experience(idx) for idx in indices]
        return [np.array(item) for item in zip(*experiences)]
