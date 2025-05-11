# -*- coding: utf-8 -*-
"""
异构图神经网络模型用于RFID定位
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, HeteroConv, Linear


class HeterogeneousGNNModel(torch.nn.Module):
    """
    用于RFID定位的异构图神经网络模型
    处理标签节点和天线节点之间的异构关系
    """

    def __init__(self, in_channels, hidden_channels, out_channels, heads=3):
        super().__init__()

        # 定义节点类型和关系类型
        self.node_types = ['tag', 'antenna']
        self.edge_types = [
            ('tag', 'to', 'tag'),  # 标签到标签的关系
            ('tag', 'to', 'antenna'),  # 标签到天线的关系
            ('antenna', 'to', 'tag')  # 天线到标签的关系
        ]

        # 异构卷积层
        self.conv1 = HeteroConv({
            ('tag', 'to', 'tag'):
                GATv2Conv(
                    in_channels,
                    hidden_channels,
                    heads=heads,
                    dropout=0.1,
                    add_self_loops=True,
                    edge_dim=1
                ),
            ('tag', 'to', 'antenna'):
                GATv2Conv(
                    in_channels,
                    hidden_channels,
                    heads=heads,
                    dropout=0.1,
                    add_self_loops=False,
                    edge_dim=1
                ),
            ('antenna', 'to', 'tag'):
                GATv2Conv(
                    in_channels,
                    hidden_channels,
                    heads=heads,
                    dropout=0.1,
                    add_self_loops=False,
                    edge_dim=1
                ),
        },
                                aggr='sum')

        self.conv2 = HeteroConv({
            ('tag', 'to', 'tag'):
                GATv2Conv(
                    hidden_channels * heads,
                    hidden_channels * heads,
                    heads=1,
                    dropout=0.1,
                    add_self_loops=True,
                    edge_dim=1
                ),
            ('tag', 'to', 'antenna'):
                GATv2Conv(
                    hidden_channels * heads,
                    hidden_channels * heads,
                    heads=1,
                    dropout=0.1,
                    add_self_loops=False,
                    edge_dim=1
                ),
            ('antenna', 'to', 'tag'):
                GATv2Conv(
                    hidden_channels * heads,
                    hidden_channels * heads,
                    heads=1,
                    dropout=0.1,
                    add_self_loops=False,
                    edge_dim=1
                ),
        },
                                aggr='sum')

        # 标签节点的MLP预测头
        self.fc = torch.nn.Sequential(
            Linear(hidden_channels * heads, hidden_channels), torch.nn.ReLU(),
            torch.nn.Dropout(0.1), Linear(hidden_channels, 32), torch.nn.ReLU(),
            Linear(32, out_channels)
        )

        # 残差连接的线性变换
        if in_channels != hidden_channels * heads:
            self.res_fc1 = Linear(in_channels, hidden_channels * heads)
        else:
            self.res_fc1 = None

        # 第二个残差连接
        self.res_fc2 = None  # 预留接口，便于后续扩展

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        """
        前向传播

        参数:
            x_dict: 节点特征字典 {node_type: features}
            edge_index_dict: 边索引字典 {edge_type: edge_indices}
            edge_attr_dict: 边属性字典 {edge_type: edge_attributes}

        返回:
            tag节点的位置预测
        """
        # 保存输入用于残差连接
        x_dict_input = {k: v for k, v in x_dict.items()}

        # 第一层异构卷积
        x_dict = self.conv1(x_dict, edge_index_dict, edge_attr_dict)

        # 第一处残差连接
        for node_type in x_dict:
            if self.res_fc1 is not None and node_type in x_dict_input:
                res1 = self.res_fc1(x_dict_input[node_type])
                x_dict[node_type] = F.elu(x_dict[node_type] + res1)
            elif node_type in x_dict_input:
                x_dict[node_type] = F.elu(x_dict[node_type] + x_dict_input[node_type])
            else:
                x_dict[node_type] = F.elu(x_dict[node_type])

        # 保存第一层输出用于第二个残差连接
        x_dict_mid = {k: v for k, v in x_dict.items()}

        # 第二层异构卷积
        x_dict = self.conv2(x_dict, edge_index_dict, edge_attr_dict)

        # 第二处残差连接
        for node_type in x_dict:
            if self.res_fc2 is not None and node_type in x_dict_mid:
                res2 = self.res_fc2(x_dict_mid[node_type])
                x_dict[node_type] = F.elu(x_dict[node_type] + res2)
            elif node_type in x_dict_mid:
                x_dict[node_type] = F.elu(x_dict[node_type] + x_dict_mid[node_type])
            else:
                x_dict[node_type] = F.elu(x_dict[node_type])

        # 只对标签节点执行最终的线性层
        output = self.fc(x_dict['tag'])

        return output
