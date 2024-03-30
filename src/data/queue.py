import numpy as np
from collections import deque


class MaxSizedQueue:
    def __init__(self, history_len=4, element_shape=(84, 84)):
        self.element_shape = element_shape
        self.history_len = history_len
        self.queue = deque(maxlen=history_len)

    def enqueue(self, element):
        self.queue.append(element)

    def get_history(self):
        if len(self.queue) == 0:
            # If queue is empty, return array filled with zeros
            return np.zeros((*self.element_shape, self.history_len), dtype=np.float32)
        elif len(self.queue) < self.history_len:
            # If queue has fewer elements than history_len
            num_missing = self.history_len - len(self.queue)
            oldest_element = self.queue[0]
            missing_elements = [oldest_element] * num_missing
            return np.concatenate([*missing_elements, *self.queue], axis=-1)
        else:
            # Otherwise, concatenate the elements in the queue
            # along the last axis
            return np.concatenate(list(self.queue), axis=-1)


