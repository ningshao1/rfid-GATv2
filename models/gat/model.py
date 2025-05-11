# -*- coding: utf-8 -*-
"""
用于定位的图注意力网络模型
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class GATLocalizationModel(torch.nn.Module):
    """用于定位的图注意力网络模型"""

    def __init__(self, in_channels, hidden_channels, out_channels, heads=1):
        super().__init__()
        self.gat1 = GATv2Conv(
            in_channels,
            hidden_channels,
            heads=heads,
            dropout=0.12,
            add_self_loops=True,
            edge_dim=1
        )
        self.gat2 = GATv2Conv(
            hidden_channels * heads,
            hidden_channels * heads,
            heads=1,
            dropout=0.12,
            add_self_loops=True,
            edge_dim=1
        )

        # 残差连接的线性变换（如果需要）
        if in_channels != hidden_channels * heads:
            self.res_fc1 = torch.nn.Linear(in_channels, hidden_channels * heads)
        else:
            self.res_fc1 = None
        # 第二个残差连接（gat2前后）
        # gat2输入输出维度都是hidden_channels * heads
        self.res_fc2 = None  # 预留接口，便于后续扩展

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * heads, hidden_channels), torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 32), torch.nn.ReLU(),
            torch.nn.Linear(32, out_channels)
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # 确保边权重是2D张量，这是GATv2Conv所需的
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.view(-1, 1)

        x_input = x  # 保留输入用于残差
        x = self.gat1(x, edge_index, edge_attr=edge_attr)
        # 第一处残差连接
        if self.res_fc1 is not None:
            res1 = self.res_fc1(x_input)
        else:
            res1 = x_input
        x = F.elu(x + res1)

        x_input2 = x  # 保留gat1输出用于第二处残差
        x = self.gat2(x, edge_index, edge_attr=edge_attr)
        # 第二处残差连接
        if self.res_fc2 is not None:
            res2 = self.res_fc2(x_input2)
        else:
            res2 = x_input2
        x = F.elu(x + res2)

        x = self.fc(x)
        return x
