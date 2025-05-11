# -*- coding: utf-8 -*-
"""
异构图神经网络的工具函数
"""

import torch
import numpy as np
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import HeteroData


def create_heterogeneous_graph_data(
    features_norm, labels_norm, antenna_positions, k=7, device='cpu'
):
    """
    创建异构图数据，包含标签节点和天线节点

    参数:
        features_norm: 标准化后的特征
        labels_norm: 标准化后的标签
        antenna_positions: 天线位置坐标
        k: KNN的K值
        device: 计算设备

    返回:
        HeteroData对象
    """
    # 创建异构数据对象
    data = HeteroData()

    # 获取节点数量
    num_tags = len(features_norm)
    num_antennas = len(antenna_positions)

    # 标准化天线位置
    if not isinstance(antenna_positions, torch.Tensor):
        antenna_positions = torch.tensor(antenna_positions, dtype=torch.float32)

    # 转移到指定设备
    features_norm = features_norm.to(device)
    labels_norm = labels_norm.to(device)
    antenna_positions = antenna_positions.to(device)

    # 为天线节点创建特征
    antenna_features = torch.full((num_antennas, features_norm.shape[1]),
                                  -1.0,
                                  dtype=torch.float32).to(device)

    # 设置节点特征和标签
    data['tag'].x = features_norm
    data['tag'].y = labels_norm
    data['antenna'].x = antenna_features
    data['antenna'].y = antenna_positions

    # 创建标签-标签之间的边（基于KNN）
    # 使用特征和位置的组合来计算KNN
    combined_features = torch.cat(
        [
            0.2 * features_norm,  # 给特征较小的权重
            0.8 * labels_norm  # 给位置较大的权重
        ],
        dim=1
    ).cpu().numpy()

    # 计算KNN邻接矩阵
    adj_matrix = kneighbors_graph(
        combined_features,
        n_neighbors=k,
        mode='distance',
    )

    # 提取边索引和边属性
    adj_matrix_coo = adj_matrix.tocoo()
    edge_index = torch.tensor(
        np.vstack([adj_matrix_coo.row, adj_matrix_coo.col]), dtype=torch.long
    ).to(device)
    edge_attr = torch.tensor(adj_matrix_coo.data,
                             dtype=torch.float32).view(-1, 1).to(device)

    # 设置标签-标签边
    data['tag', 'to', 'tag'].edge_index = edge_index
    data['tag', 'to', 'tag'].edge_attr = edge_attr

    # 创建标签-天线和天线-标签之间的边
    tag_to_antenna_edges = []
    antenna_to_tag_edges = []
    tag_to_antenna_attrs = []
    antenna_to_tag_attrs = []

    # 为每个标签节点添加到各个天线的边
    for tag_idx in range(num_tags):
        for antenna_idx in range(num_antennas):
            # 计算标签节点到天线的欧氏距离
            pos_dist = torch.sqrt(
                torch.sum((labels_norm[tag_idx] - antenna_positions[antenna_idx])**2)
            )

            # 特征距离
            feat_weight = 0.3  # 特征权重
            dist = (1 - feat_weight) * pos_dist + feat_weight * \
                torch.norm(features_norm[tag_idx])

            # 添加双向边
            tag_to_antenna_edges.append([tag_idx, antenna_idx])
            antenna_to_tag_edges.append([antenna_idx, tag_idx])

            # 边属性（距离）
            tag_to_antenna_attrs.append(dist.item())
            antenna_to_tag_attrs.append(dist.item())

    # 转换为张量并设置边
    if tag_to_antenna_edges:
        tag_to_antenna_edge_index = torch.tensor(
            tag_to_antenna_edges, dtype=torch.long
        ).t().to(device)
        antenna_to_tag_edge_index = torch.tensor(
            antenna_to_tag_edges, dtype=torch.long
        ).t().to(device)

        tag_to_antenna_edge_attr = torch.tensor(
            tag_to_antenna_attrs, dtype=torch.float32
        ).view(-1, 1).to(device)
        antenna_to_tag_edge_attr = torch.tensor(
            antenna_to_tag_attrs, dtype=torch.float32
        ).view(-1, 1).to(device)

        data['tag', 'to', 'antenna'].edge_index = tag_to_antenna_edge_index
        data['tag', 'to', 'antenna'].edge_attr = tag_to_antenna_edge_attr

        data['antenna', 'to', 'tag'].edge_index = antenna_to_tag_edge_index
        data['antenna', 'to', 'tag'].edge_attr = antenna_to_tag_edge_attr

    return data


def add_new_node_to_hetero_graph(
    hetero_data, new_features, initial_position, k=5, device='cpu'
):
    """
    向已有的异构图中添加新节点

    参数:
        hetero_data: 已有的异构图数据
        new_features: 新节点的特征
        initial_position: 新节点的初始位置估计
        k: KNN的K值
        device: 计算设备

    返回:
        更新后的HeteroData对象
    """
    # 创建新数据对象
    new_data = HeteroData()

    # 获取原有标签节点和天线节点
    original_tag_features = hetero_data['tag'].x
    original_tag_positions = hetero_data['tag'].y

    antenna_features = hetero_data['antenna'].x
    antenna_positions = hetero_data['antenna'].y

    # 添加新节点
    num_original_tags = len(original_tag_features)

    # 确保数据类型和维度一致
    if not isinstance(new_features, torch.Tensor):
        new_features = torch.tensor(new_features, dtype=torch.float32)
    if new_features.dim() == 1:
        new_features = new_features.unsqueeze(0)

    if not isinstance(initial_position, torch.Tensor):
        initial_position = torch.tensor(initial_position, dtype=torch.float32)
    if initial_position.dim() == 1:
        initial_position = initial_position.unsqueeze(0)

    # 转移到指定设备
    new_features = new_features.to(device)
    initial_position = initial_position.to(device)

    # 合并标签节点
    all_tag_features = torch.cat([original_tag_features, new_features], dim=0)
    all_tag_positions = torch.cat([original_tag_positions, initial_position], dim=0)

    # 设置节点特征和标签
    new_data['tag'].x = all_tag_features
    new_data['tag'].y = all_tag_positions
    new_data['antenna'].x = antenna_features
    new_data['antenna'].y = antenna_positions

    # 添加tag_mask来标记新节点
    tag_mask = torch.zeros(len(all_tag_features), dtype=torch.bool, device=device)
    tag_mask[num_original_tags:] = True  # 将新添加的节点标记为True
    new_data['tag'].tag_mask = tag_mask

    # 创建标签-标签之间的边（基于KNN）
    # 使用特征和位置的组合来计算KNN
    combined_features = torch.cat(
        [
            0.2 * all_tag_features,  # 给特征较小的权重
            0.8 * all_tag_positions  # 给位置较大的权重
        ],
        dim=1
    ).cpu().numpy()

    # 计算KNN邻接矩阵
    adj_matrix = kneighbors_graph(
        combined_features,
        n_neighbors=k,
        mode='distance',
    )

    # 提取边索引和边属性
    adj_matrix_coo = adj_matrix.tocoo()
    edge_index = torch.tensor(
        np.vstack([adj_matrix_coo.row, adj_matrix_coo.col]), dtype=torch.long
    ).to(device)
    edge_attr = torch.tensor(adj_matrix_coo.data,
                             dtype=torch.float32).view(-1, 1).to(device)

    # 设置标签-标签边
    new_data['tag', 'to', 'tag'].edge_index = edge_index
    new_data['tag', 'to', 'tag'].edge_attr = edge_attr

    # 创建标签-天线和天线-标签之间的边
    num_antennas = len(antenna_positions)
    tag_to_antenna_edges = []
    antenna_to_tag_edges = []
    tag_to_antenna_attrs = []
    antenna_to_tag_attrs = []

    # 为每个标签节点添加到各个天线的边
    for tag_idx in range(len(all_tag_features)):
        for antenna_idx in range(num_antennas):
            # 计算标签节点到天线的欧氏距离
            pos_dist = torch.sqrt(
                torch.sum(
                    (all_tag_positions[tag_idx] - antenna_positions[antenna_idx])**2
                )
            )

            # 特征距离
            feat_weight = 0.3  # 特征权重
            dist = (1 - feat_weight) * pos_dist + feat_weight * \
                torch.norm(all_tag_features[tag_idx])

            # 添加双向边
            tag_to_antenna_edges.append([tag_idx, antenna_idx])
            antenna_to_tag_edges.append([antenna_idx, tag_idx])

            # 边属性（距离）
            tag_to_antenna_attrs.append(dist.item())
            antenna_to_tag_attrs.append(dist.item())

    # 转换为张量并设置边
    if tag_to_antenna_edges:
        tag_to_antenna_edge_index = torch.tensor(
            tag_to_antenna_edges, dtype=torch.long
        ).t().to(device)
        antenna_to_tag_edge_index = torch.tensor(
            antenna_to_tag_edges, dtype=torch.long
        ).t().to(device)

        tag_to_antenna_edge_attr = torch.tensor(
            tag_to_antenna_attrs, dtype=torch.float32
        ).view(-1, 1).to(device)
        antenna_to_tag_edge_attr = torch.tensor(
            antenna_to_tag_attrs, dtype=torch.float32
        ).view(-1, 1).to(device)

        new_data['tag', 'to', 'antenna'].edge_index = tag_to_antenna_edge_index
        new_data['tag', 'to', 'antenna'].edge_attr = tag_to_antenna_edge_attr

        new_data['antenna', 'to', 'tag'].edge_index = antenna_to_tag_edge_index
        new_data['antenna', 'to', 'tag'].edge_attr = antenna_to_tag_edge_attr

    return new_data
