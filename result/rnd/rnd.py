import torch.nn as nn
import torch

class RandomQNetwork(nn.Module):
    def __init__(self, observation_shape, action_shape, device='cpu'):
        super().__init__()
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.device = device

    def forward(self, obs):
        return torch.rand((1, self.action_shape)).to(self.device)