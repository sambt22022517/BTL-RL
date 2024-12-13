import torch.nn as nn
import torch


class ConvResidual(nn.Module):
    def __init__(self, channels, device='cpu'):
        super().__init__()
        self.channels = channels
        self.device = device

        self.model = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1)
        ).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        model_result = self.model(x)
        return x + model_result


class LinearResidual(nn.Module):
    def __init__(self, size, device='cpu'):
        super().__init__()
        self.size = size
        self.device = device

        self.model = nn.Sequential(
            nn.Linear(self.size, self.size),
            nn.ReLU(),
            nn.Linear(self.size, self.size)
        ).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        model_result = self.model(x)
        return x + model_result
        

class QNetwork(nn.Module):
    def __init__(self, observation_shape, action_shape, device='cpu'):
        super().__init__()
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.device = device

        self.cnn = nn.Sequential(
            nn.Conv2d(observation_shape[-1], 16, 3, padding=1),
            ConvResidual(16, self.device),
            ConvResidual(16, self.device),
            ConvResidual(16, self.device),
        ).to(self.device)
        
        self.network = nn.Sequential(
            nn.Linear(observation_shape[0] * observation_shape[1] * 16, 120),
            LinearResidual(120, self.device),
            LinearResidual(120, self.device),
            LinearResidual(120, self.device),
            nn.Linear(120, action_shape),
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
