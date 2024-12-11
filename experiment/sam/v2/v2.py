import torch.nn as nn
import torch


class QNetwork(nn.Module):
    def __init__(self, observation_shape, action_shape, device='cpu'):
        super().__init__()
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.device = device

        self.cnn = nn.Sequential(
            nn.Conv2d(observation_shape[-1], observation_shape[-1], 3),
            nn.ReLU(),
            nn.Conv2d(observation_shape[-1], observation_shape[-1], 3),
            nn.ReLU(),
        ).to(self.device)

        dummy_input = torch.randn(observation_shape).permute(2, 0, 1).to(self.device)
        dummy_output = self.cnn(dummy_input)
        flatten_dim = dummy_output.view(-1).shape[0]
        self.network = nn.Sequential(
            nn.Linear(flatten_dim, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, action_shape),
        ).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        assert len(x.shape) >= 3, "only support magent input observation"
        x = self.cnn(x)
        if len(x.shape) == 3:
            batchsize = 1
        else:
            batchsize = x.shape[0]
        x = x.reshape(batchsize, -1)
        return self.network(x)
