import numpy as np
import torch


class BufferReplay:
    def __init__(self, max_capacity, state_dim, action_dim):

        self.rng = np.random.default_rng()

        self.ptr = 0
        self.size = 0
        self.max_capacity = max_capacity

        self.states = np.zeros((max_capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_capacity, action_dim), dtype=np.float32)

        self.rewards = np.zeros(max_capacity, dtype=np.float32)
        self.new_states = np.zeros((max_capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(max_capacity, dtype=np.float32)

    def sample(self, sample_size):

        if self.size < sample_size:
            raise ValueError("Not enough samples in the buffer to sample from.")

        indices = self.rng.integers(0, self.size, size=sample_size)

        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        new_states = self.new_states[indices]
        dones = self.dones[indices]

        def t(x):
            return torch.tensor(x, dtype=torch.float32)

        return t(states), t(actions), t(rewards), t(new_states), t(dones)

    def store(self, state, action, reward, new_state, done):

        i = self.ptr

        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.new_states[i] = new_state
        self.dones[i] = done

        self.ptr = (self.ptr + 1) % self.max_capacity
        self.size = min(self.size + 1, len(self.states))

    def __len__(self):
        return self.size
