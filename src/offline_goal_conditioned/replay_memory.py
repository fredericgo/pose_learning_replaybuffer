import random
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader


class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, goal, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, goal, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, goal, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, goal, done

    def __len__(self):
        return len(self.buffer)

    def save(self, memory_path):
        data = dict()

        data['state'] = np.array([x[0] for x in self.buffer], dtype=np.float32)
        data['action'] = np.array([x[1] for x in self.buffer], dtype=np.float32)
        data['reward'] = np.array([x[2] for x in self.buffer], dtype=np.float32)
        data['next_state'] = np.array([x[3] for x in self.buffer], dtype=np.float32)
        data['goal'] = np.array([x[4] for x in self.buffer], dtype=np.float32)
        data['done'] = np.array([x[5] for x in self.buffer], dtype=np.float32)

        torch.save(data, memory_path)

