import torch
import torch.nn as nn

class MyQNetwork(nn.Module):
    def __init__(self, observation_shape, action_shape):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(observation_shape[-1], 13, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(13, 13, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(13, 13, kernel_size=3),
            nn.ReLU(),
        )
        
        dummy_input = torch.randn(observation_shape).permute(2, 0, 1)
        dummy_output = self.cnn(dummy_input)
        flatten_dim = dummy_output.view(-1).shape[0]
        
        self.fc = nn.Sequential(
            nn.Linear(flatten_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            # nn.Linear(512, 256),
            # nn.ReLU(),
            # nn.Linear(256, 128),
            # nn.ReLU(),
            nn.Linear(128, action_shape)
        )

    def forward(self, x):
        assert len(x.shape) >= 3, "only support magent input observation"
        out = self.cnn(x)

        # print(out.shape)
        
        # out += self.skip_connection(x)

        # out = self.cnn2(out)
        
        if len(x.shape) == 3:
            batchsize = 1
            
        else:
            batchsize = x.shape[0]
            
        out = out.reshape(batchsize, -1)
        
        return self.fc(out)
