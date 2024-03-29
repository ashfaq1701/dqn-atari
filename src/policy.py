import numpy as np

def epsilon_greedy_policy(state_history, model, n_outputs, epsilon=0):
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs)
    else:
        Q_values = model.predict(state_history[np.newaxis], verbose=0)[0]
        return Q_values.argmax()