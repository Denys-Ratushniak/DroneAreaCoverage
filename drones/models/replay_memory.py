import torch
import random
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, next_state, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, next_state, reward)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))

        batch_state_tensor = torch.cat(batch.state).reshape(batch_size, -1, 6)
        batch_action_tensor = torch.cat(batch.action).reshape(batch_size, -1, 2)
        batch_next_state_tensor = torch.cat(batch.next_state).reshape(batch_size, -1, 6)
        batch_reward_tensor = torch.tensor(batch.reward).view(-1, 1)

        return batch_state_tensor, batch_action_tensor, batch_next_state_tensor, batch_reward_tensor

    def __len__(self):
        return len(self.memory)

