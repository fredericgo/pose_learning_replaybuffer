import random
import numpy as np

import torch


class EpisodicReplayBuffer(object):
    """
    Wrapper around a replay buffer in order to use Episodic replay buffer.
    :param replay_buffer: (ReplayBuffer)
    """

    def __init__(self, capacity, seed):
        super(EpisodicReplayBuffer, self).__init__()
        # Buffer for storing transitions of the current episode
        random.seed(seed)

        self._buffer = []
        self.episode_lengths = []
        self.capacity = capacity
        self.position = 0
    
    def add_episode(self, states, actions, rewards, next_states, dones):
        # For each transition in the last episode,
        # create a set of artificial transitions
        if len(self._buffer) < self.capacity:
            self._buffer.append(None)
            self.episode_lengths.append(None)
        self._buffer[self.position] = (states, actions, rewards, next_states, dones)
        self.episode_lengths[self.position] = len(states)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, horizon=None):
    
        # sample episodes
        episode_idx = np.random.choice(np.arange(len(self._buffer)), batch_size)

        episode_len = self._buffer[0][0].shape[0]

        if not horizon:
            horizon = 0

        state_idx = np.random.randint(0, episode_len - horizon, batch_size)

        high = state_idx + horizon if horizon > 0 else episode_len
        goal_idx = np.random.randint(state_idx, high, batch_size)
        
        state = np.array([self._buffer[e][0][s] for e, s in zip(episode_idx, state_idx)], dtype=np.float32)
        action = np.array([self._buffer[e][1][s] for e, s in zip(episode_idx, state_idx)], dtype=np.float32)
        goal = np.array([self._buffer[e][0][g] for e, g in zip(episode_idx, goal_idx)], dtype=np.float32)
        next_state = np.array([self._buffer[e][3][s] for e, s in zip(episode_idx, state_idx)], dtype=np.float32)
        
        done = np.array(state_idx == goal_idx, dtype=np.float32)
        mask = 1 - done
        reward = done
        
        return state, action, goal, reward, next_state, mask

    
    def sample_state_and_goal(self, batch_size, horizon=0):
    
        # sample episodes
        episode_idx = np.random.choice(np.arange(len(self._buffer)), batch_size)
        episode_len = self._buffer[0][0].shape[0]
   
        state_idx = np.random.randint(0, episode_len - horizon, batch_size)
        goal_idx = np.random.randint(state_idx, state_idx + horizon if horizon > 0 else episode_len, batch_size)

        state = np.array([self._buffer[e][0][s] for e, s in zip(episode_idx, state_idx)], dtype=np.float32)
        goal = np.array([self._buffer[e][0][g] for e, g in zip(episode_idx, goal_idx)], dtype=np.float32)
    
        return state, goal
  
    def __len__(self):
        return len(self._buffer)

    def save(self, memory_path):
        data = dict()

        data['state'] = np.array([x[0] for x in self._buffer], dtype=np.float32)
        data['action'] = np.array([x[1] for x in self._buffer], dtype=np.float32)
        data['reward'] = np.array([x[2] for x in self._buffer], dtype=np.float32)
        data['next_state'] = np.array([x[3] for x in self._buffer], dtype=np.float32)
        data['done'] = np.array([x[4] for x in self._buffer], dtype=np.float32)
        torch.save(data, memory_path)

    def load(self, memory_path):
        data = torch.load(memory_path)
        data['done'] = data['done'].astype(bool)
        n_episodes = data['state'].shape[0]
        for e in range(n_episodes):
            self.add_episode(data['state'][e],
                             data['action'][e],
                             data['reward'][e],
                             data['next_state'][e],
                             data['done'][e])

    def as_dataset(self):
        data = dict()

        data['state'] = np.array([x[0] for x in self._buffer], dtype=np.float32)
        data['action'] = np.array([x[1] for x in self._buffer], dtype=np.float32)
        data['reward'] = np.array([x[2] for x in self._buffer], dtype=np.float32)
        data['next_state'] = np.array([x[3] for x in self._buffer], dtype=np.float32)
        data['done'] = np.array([x[4] for x in self._buffer], dtype=np.float32)
        return data