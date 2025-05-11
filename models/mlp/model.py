# -*- coding: utf-8 -*-
"""
用于定位的MLP网络模型
"""

import torch
import torch.nn.functional as F


class MLPLocalizationModel(torch.nn.Module):
    """用于定位的MLP网络模型"""

    def __init__(self, in_channels, hidden_channels=128, out_channels=2, dropout=0.1):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_channels, hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.fc3 = torch.nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc4 = torch.nn.Linear(hidden_channels // 2, out_channels)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
