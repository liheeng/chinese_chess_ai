import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += residual
        return F.relu(out)


class ChineseChessNet(nn.Module):
    def __init__(self, in_channels=26, hidden=192, num_blocks=10, action_space=8100):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(),
        )
        self.res_blocks = nn.ModuleList([ResBlock(hidden) for _ in range(num_blocks)])
        # Policy head: Transformer
        self.policy_conv = nn.Conv2d(hidden, 32, 1)
        self.policy_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=32 * 10 * 9, nhead=8, batch_first=True),
            num_layers=2,
        )
        self.policy_fc = nn.Linear(32 * 10 * 9, action_space)
        # Value head
        self.value_conv = nn.Conv2d(hidden, 3, 1)
        self.value_fc = nn.Sequential(
            nn.Linear(3 * 10 * 9, 128), nn.ReLU(), nn.Linear(128, 1), nn.Tanh()
        )

    def forward(self, x):
        # x: (B, 72, 10, 9)
        x = self.conv_in(x)
        for block in self.res_blocks:
            x = block(x)
        # Policy
        p = self.policy_conv(x)
        p = p.view(p.size(0), -1).unsqueeze(1)  # (B, 1, 32*10*9)
        p = self.policy_transformer(p).squeeze(1)
        policy = self.policy_fc(p)  # (B, 8100)
        # Value
        v = self.value_conv(x)
        v = v.view(v.size(0), -1)
        value = self.value_fc(v).squeeze(-1)
        return policy, value
