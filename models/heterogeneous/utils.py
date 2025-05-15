# -*- coding: utf-8 -*-
"""
异构图神经网络的工具函数
"""

import torch
import numpy as np
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import HeteroData
import sys
import os

# 添加项目根目录到系统路径
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from config import CONFIG

# 设置随机种子，从config.py中获取RANDOM_SEED
RANDOM_SEED = CONFIG['RANDOM_SEED']
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)


def check_obstruction(antenna_pos, tag_pos, chair_pos, chair_size):
    """
    检测椅子是否位于天线和标签之间，形成遮挡

    参数:
        antenna_pos: 天线位置坐标
        tag_pos: 标签位置坐标
        chair_pos: 椅子位置坐标
        chair_size: 椅子大小

    返回:
        is_obstructing: 布尔值，表示是否存在遮挡
        line_distance: 椅子到天线-标签连线的距离
    """
    # 将所有输入转换为numpy数组以便计算
    if isinstance(antenna_pos, torch.Tensor):
        antenna_pos = antenna_pos.cpu().numpy()
    if isinstance(tag_pos, torch.Tensor):
        tag_pos = tag_pos.cpu().numpy()
    if isinstance(chair_pos, torch.Tensor):
        chair_pos = chair_pos.cpu().numpy()

    # 检查天线和标签位置是否重合
    if np.array_equal(antenna_pos, tag_pos):
        # 如果重合，计算椅子到点的距离
        point_distance = np.sqrt(np.sum((chair_pos - antenna_pos)**2))
        # 当椅子与天线/标签重合点的距离小于椅子大小时，认为存在遮挡
        is_obstructing = point_distance < chair_size
        return is_obstructing, point_distance

    # 计算椅子到天线-标签连线的距离
    A = tag_pos[1] - antenna_pos[1]
    B = antenna_pos[0] - tag_pos[0]
    C = tag_pos[0] * antenna_pos[1] - antenna_pos[0] * tag_pos[1]

    # 避免除以零错误
    denominator = np.sqrt(A**2 + B**2)
    if denominator < 1e-10:  # 设置一个很小的阈值
        # 天线和标签位置几乎重合，计算椅子到点的距离
        point_distance = np.sqrt(np.sum((chair_pos - antenna_pos)**2))
        is_obstructing = point_distance < chair_size
        return is_obstructing, point_distance

    line_distance = abs(A * chair_pos[0] + B * chair_pos[1] + C) / denominator

    # 检查椅子是否在天线和标签之间的范围内
    min_x = min(antenna_pos[0], tag_pos[0])
    max_x = max(antenna_pos[0], tag_pos[0])
    min_y = min(antenna_pos[1], tag_pos[1])
    max_y = max(antenna_pos[1], tag_pos[1])

    in_range = (
        min_x - chair_size <= chair_pos[0] <= max_x + chair_size
        and min_y - chair_size <= chair_pos[1] <= max_y + chair_size
    )

    # 判断是否存在遮挡：距离小于椅子大小的2倍且在范围内
    is_obstructing = line_distance < chair_size * 2 and in_range

    return is_obstructing, line_distance


def create_heterogeneous_graph_data(
    features_norm,
    labels_norm,
    antenna_positions,
    k=7,
    device='cpu',
    chair_info=None,
    edge_attr_weights=None
):
    """
    创建异构图数据，包含标签节点、天线节点和椅子节点

    参数:
        features_norm: 标准化后的特征
        labels_norm: 标准化后的标签
        antenna_positions: 天线位置坐标
        k: KNN的K值
        device: 计算设备
        chair_info: 椅子信息列表，每个元素为包含类型、位置、大小和材质信息的字典
                    例如：[{'type': '椅子', 'position': (6, 6), 'size': 0.8, 'material': 'wood'}]
        edge_attr_weights: 边属性计算的权重参数字典，包含：
                           {'w1': 0.6, 'w2': 0.1, 'w3': 0.3}
                           w1: 距离影响权重
                           w2: RSSI影响权重
                           w3: 材质和大小影响权重

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
            feat_weight = 0.2  # 特征权重
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

    # 添加椅子节点（如果提供了椅子信息）
    if chair_info is not None:
        # 确保chair_info是列表类型
        if not isinstance(chair_info, list):
            chair_info = [chair_info]

        num_chairs = len(chair_info)

        # 如果没有椅子信息，直接返回
        if num_chairs == 0:
            return data

        # 椅子特征向量的维度与标签特征维度相同
        chair_feature_dim = features_norm.shape[1]
        chair_features = torch.zeros((num_chairs, chair_feature_dim),
                                     dtype=torch.float32).to(device)
        chair_positions = []

        # 为每个椅子计算特征和位置
        for i, chair in enumerate(chair_info):
            # 获取椅子信息
            chair_position = chair.get('position', (0, 0))
            chair_size = chair.get('size', 1.0)
            chair_material = chair.get('material', 'wood')

            # 将椅子位置转换为张量
            if not isinstance(chair_position, torch.Tensor):
                chair_position_tensor = torch.tensor(
                    chair_position, dtype=torch.float32
                ).to(device)
            else:
                chair_position_tensor = chair_position.to(device)

            if chair_position_tensor.dim() == 1:
                chair_position_tensor = chair_position_tensor.unsqueeze(0)

            chair_positions.append(chair_position_tensor)

            # 计算椅子的物理特性
            # 材质对电磁波的影响系数（与仿真代码一致）
            material_coef = {
                'wood': 1.5,
                'metal': 4.0,
                'plastic': 0.1,
                'electronic': 3.5  # 电子设备也有较强的信号干扰
            }.get(chair_material, 0.2)

            # 前两个维度设置为椅子的物理特性
            chair_features[i, 0] = chair_size  # 第一维为椅子大小
            chair_features[i, 1] = material_coef  # 第二维为材质系数

        # 将所有椅子位置合并为一个张量
        if chair_positions:
            chair_positions_tensor = torch.cat(chair_positions, dim=0)
            # 设置椅子节点特征和位置
            data['chair'].x = chair_features
            data['chair'].y = chair_positions_tensor

            # 创建标签-椅子和椅子-标签之间的边
            tag_to_chair_edges = []
            chair_to_tag_edges = []
            tag_to_chair_attrs = []
            chair_to_tag_attrs = []

            # 为每个标签节点添加到每个椅子的边，仅当存在遮挡关系时
            for tag_idx in range(num_tags):
                for chair_idx in range(num_chairs):
                    # 检查是否存在遮挡关系
                    has_obstruction = False
                    obstruction_factor = 0.0
                    chair_size = chair_features[chair_idx, 0].item()
                    material_coef = chair_features[chair_idx, 1].item()

                    # 检查是否有天线到标签的信号被椅子遮挡
                    for antenna_idx in range(num_antennas):
                        antenna_pos = antenna_positions[antenna_idx]
                        # 使用遮挡检测函数
                        is_obstructing, line_distance = check_obstruction(
                            antenna_pos, labels_norm[tag_idx],
                            chair_positions_tensor[chair_idx], chair_size
                        )
                        if is_obstructing:
                            has_obstruction = True
                            # 遮挡强度计算
                            blockage_strength = material_coef * (
                                1.0 - line_distance / (chair_size * 2.0)
                            )
                            obstruction_factor += blockage_strength

                    # 只有当存在遮挡关系时才建立边
                    if has_obstruction:
                        # 计算标签节点到椅子的欧氏距离
                        pos_dist = torch.sqrt(
                            torch.sum((
                                labels_norm[tag_idx] - chair_positions_tensor[chair_idx]
                            )**2)
                        )

                        # 参数设置
                        w1 = edge_attr_weights['w1']  # 距离影响权重
                        w2 = edge_attr_weights['w2']  # RSSI影响权重
                        w3 = edge_attr_weights['w3']  # 材质和大小影响权重

                        # 1. 距离项: 1/(1+d)
                        distance_factor = 1.0 / (1.0 + pos_dist.item())

                        # 2. RSSI均值项
                        # 假设RSSI值存储在features的前4个维度
                        if features_norm.shape[1] >= 4:
                            rssi_values = features_norm[tag_idx, :4]
                            rssi_mean = torch.mean(rssi_values).item()
                        else:
                            # 如果没有足够的RSSI值，使用特征的平均值
                            rssi_mean = torch.mean(features_norm[tag_idx]).item()

                        # 3. 材质与大小的衰减项
                        # L_chair代表材质系数，size代表椅子大小
                        L_chair = material_coef
                        if pos_dist.item() > 0:
                            decay_factor = L_chair * torch.exp(
                                torch.tensor(-chair_size / (2 * pos_dist.item()))
                            ).item()
                        else:
                            # 避免除以零
                            decay_factor = L_chair * torch.exp(
                                torch.tensor(-chair_size / 0.01)
                            ).item()

                        # 4. 多径效应影响
                        multipath_factor = 0.0
                        if pos_dist.item() > 0:
                            multipath_factor = 0.2 * L_chair * np.sin(
                                pos_dist.item() * 5
                            )

                        # 5. 反射与散射效应（特别是金属材质）
                        reflection_factor = 0.0
                        if L_chair > 3:  # 金属材质
                            reflection_factor = 0.3 * np.exp(-pos_dist.item() / 3)

                        # 组合所有项
                        physics_effect = (
                            w1 * distance_factor + w2 * rssi_mean + w3 * (
                                decay_factor + multipath_factor + reflection_factor +
                                1.5 * obstruction_factor
                            )
                        )

                        # 添加双向边
                        tag_to_chair_edges.append([tag_idx, chair_idx])
                        chair_to_tag_edges.append([chair_idx, tag_idx])

                        # 边属性（物理效应）
                        tag_to_chair_attrs.append(physics_effect)
                        chair_to_tag_attrs.append(physics_effect)

            # 转换为张量并设置边
            if tag_to_chair_edges:
                tag_to_chair_edge_index = torch.tensor(
                    tag_to_chair_edges, dtype=torch.long
                ).t().to(device)
                chair_to_tag_edge_index = torch.tensor(
                    chair_to_tag_edges, dtype=torch.long
                ).t().to(device)

                tag_to_chair_edge_attr = torch.tensor(
                    tag_to_chair_attrs, dtype=torch.float32
                ).view(-1, 1).to(device)
                chair_to_tag_edge_attr = torch.tensor(
                    chair_to_tag_attrs, dtype=torch.float32
                ).view(-1, 1).to(device)

                data['tag', 'to', 'chair'].edge_index = tag_to_chair_edge_index
                data['tag', 'to', 'chair'].edge_attr = tag_to_chair_edge_attr

                data['chair', 'to', 'tag'].edge_index = chair_to_tag_edge_index
                data['chair', 'to', 'tag'].edge_attr = chair_to_tag_edge_attr

            # 创建椅子-天线和天线-椅子之间的边
            chair_to_antenna_edges = []
            antenna_to_chair_edges = []
            chair_to_antenna_attrs = []
            antenna_to_chair_attrs = []

            # 为每个椅子添加到各个天线的边
            for chair_idx in range(num_chairs):
                # 获取椅子信息
                chair_size = chair_features[chair_idx, 0].item()
                material_coef = chair_features[chair_idx, 1].item()

                for antenna_idx in range(num_antennas):
                    # 计算椅子到天线的欧氏距离
                    pos_dist = torch.sqrt(
                        torch.sum((
                            chair_positions_tensor[chair_idx] -
                            antenna_positions[antenna_idx]
                        )**2)
                    )

                    # 应用新的物理公式：椅子对天线信号的影响
                    dca = pos_dist.item()
                    L_chair = material_coef

                    # 基础边属性计算公式: chair-to-antenna = 20log10(dca) + Lchair ⋅ (dca/size)
                    if dca > 0:
                        base_attr = 20 * np.log10(dca) + L_chair * (dca / chair_size)
                    else:
                        # 避免log(0)错误，使用一个很小的值
                        base_attr = 20 * np.log10(0.01) + L_chair * (0.01 / chair_size)

                    # 添加多径效应影响
                    multipath_factor = 0.2 * L_chair * np.sin(dca * 5)

                    # 添加反射与散射效应（特别是金属材质）
                    reflection_factor = 0
                    if L_chair > 3:  # 金属材质
                        reflection_factor = 0.3 * np.exp(-dca / 3)

                    # 检查信号遮挡效应
                    obstruction_factor = 0.0
                    # 检查是否有天线到标签的信号被椅子遮挡
                    for tag_idx in range(len(labels_norm)):
                        tag_pos = labels_norm[tag_idx]
                        # 使用遮挡检测函数
                        is_obstructing, line_distance = check_obstruction(
                            antenna_positions[antenna_idx], tag_pos,
                            chair_positions_tensor[chair_idx], chair_size
                        )
                        if is_obstructing:
                            # 遮挡强度计算
                            blockage_strength = L_chair * (
                                1.0 - line_distance / (chair_size * 2.0)
                            )
                            obstruction_factor += blockage_strength

                    # 组合所有项
                    physics_effect = (
                        base_attr + multipath_factor + reflection_factor +
                        1.5 * obstruction_factor
                    )

                    # 添加双向边
                    chair_to_antenna_edges.append([chair_idx, antenna_idx])
                    antenna_to_chair_edges.append([antenna_idx, chair_idx])

                    # 边属性（物理效应）
                    chair_to_antenna_attrs.append(physics_effect)
                    antenna_to_chair_attrs.append(physics_effect)

            # 转换为张量并设置边
            if chair_to_antenna_edges:
                chair_to_antenna_edge_index = torch.tensor(
                    chair_to_antenna_edges, dtype=torch.long
                ).t().to(device)
                antenna_to_chair_edge_index = torch.tensor(
                    antenna_to_chair_edges, dtype=torch.long
                ).t().to(device)

                chair_to_antenna_edge_attr = torch.tensor(
                    chair_to_antenna_attrs, dtype=torch.float32
                ).view(-1, 1).to(device)
                antenna_to_chair_edge_attr = torch.tensor(
                    antenna_to_chair_attrs, dtype=torch.float32
                ).view(-1, 1).to(device)

                data['chair', 'to', 'antenna'].edge_index = chair_to_antenna_edge_index
                data['chair', 'to', 'antenna'].edge_attr = chair_to_antenna_edge_attr

                data['antenna', 'to', 'chair'].edge_index = antenna_to_chair_edge_index
                data['antenna', 'to', 'chair'].edge_attr = antenna_to_chair_edge_attr

            # 创建遮挡关系边：检测椅子是否遮挡天线与标签之间的信号
            chair_obstructs_tag_edges = []
            chair_obstructs_tag_attrs = []
            obstruction_counts = {}  # 记录每个标签被遮挡的次数

            # 初始化遮挡计数
            for tag_idx in range(len(labels_norm)):
                obstruction_counts[tag_idx] = 0

            # 检查每个(椅子, 标签, 天线)组合是否存在遮挡
            for tag_idx in range(len(labels_norm)):
                tag_pos = labels_norm[tag_idx]

                for chair_idx in range(num_chairs):
                    chair_pos = chair_positions_tensor[chair_idx]
                    chair_size = chair_features[chair_idx, 0].item()
                    material_coef = chair_features[chair_idx, 1].item()

                    # 检查是否存在遮挡（任何天线到该标签的路径被遮挡）
                    for antenna_idx in range(num_antennas):
                        antenna_pos = antenna_positions[antenna_idx]

                        # 使用遮挡检测函数
                        is_obstructing, line_distance = check_obstruction(
                            antenna_pos, tag_pos, chair_pos, chair_size
                        )

                        if is_obstructing:
                            # 计算遮挡强度：基于材质、距离和椅子大小
                            blockage_strength = material_coef * (
                                1.0 - line_distance / (chair_size * 2.0)
                            )

                            # 添加遮挡边
                            chair_obstructs_tag_edges.append([chair_idx, tag_idx])
                            chair_obstructs_tag_attrs.append(blockage_strength)

                            # 记录遮挡次数
                            obstruction_counts[tag_idx] += 1

                            # 找到一个遮挡就足够了，不需要继续检查其他天线
                            break

            # 设置遮挡关系边
            if chair_obstructs_tag_edges:
                # 转换为张量
                chair_obstructs_tag_edge_index = torch.tensor(
                    chair_obstructs_tag_edges, dtype=torch.long
                ).t().to(device)
                chair_obstructs_tag_edge_attr = torch.tensor(
                    chair_obstructs_tag_attrs, dtype=torch.float32
                ).view(-1, 1).to(device)

                # 添加到异构图
                data['chair', 'obstructs',
                     'tag'].edge_index = chair_obstructs_tag_edge_index
                data['chair', 'obstructs',
                     'tag'].edge_attr = chair_obstructs_tag_edge_attr

            # 为每个标签添加遮挡计数作为特征
            obstruction_feature = torch.zeros(len(labels_norm), 1,
                                              dtype=torch.float32).to(device)
            for tag_idx, count in obstruction_counts.items():
                obstruction_feature[tag_idx, 0] = count

            # 添加遮挡特征到标签节点
            if 'obstruction_count' not in data['tag']:
                data['tag'].obstruction_count = obstruction_feature

    return data


def add_new_node_to_hetero_graph(
    hetero_data,
    new_features,
    initial_position,
    k=5,
    device='cpu',
    edge_attr_weights=None
):
    """
    向已有的异构图中添加新节点

    参数:
        hetero_data: 已有的异构图数据
        new_features: 新节点的特征
        initial_position: 新节点的初始位置估计
        k: KNN的K值
        device: 计算设备
        edge_attr_weights: 边属性计算的权重参数字典，包含：
                           {'w1': 0.6, 'w2': 0.1, 'w3': 0.3}

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

    # 检查是否存在椅子节点
    has_chair = False
    try:
        # 尝试访问椅子节点特征
        chair_features = hetero_data['chair'].x
        chair_positions = hetero_data['chair'].y
        num_chairs = len(chair_positions)
        has_chair = True
    except (KeyError, AttributeError):
        # 如果发生KeyError或AttributeError，说明没有椅子节点
        has_chair = False

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

    # 如果存在椅子节点，添加到新图
    if has_chair:
        new_data['chair'].x = chair_features
        new_data['chair'].y = chair_positions

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
            feat_weight = 0.2  # 特征权重
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

    # 如果存在椅子节点，创建标签-椅子和椅子-标签之间的边
    if has_chair:
        tag_to_chair_edges = []
        chair_to_tag_edges = []
        tag_to_chair_attrs = []
        chair_to_tag_attrs = []

        # 为每个标签节点添加到每个椅子的边，仅当存在遮挡关系时
        for tag_idx in range(len(all_tag_features)):
            for chair_idx in range(num_chairs):
                # 检查是否存在遮挡关系
                has_obstruction = False
                obstruction_factor = 0.0
                chair_size = chair_features[chair_idx, 0].item()
                material_coef = chair_features[chair_idx, 1].item()

                # 检查是否有天线到标签的信号被椅子遮挡
                for antenna_idx in range(num_antennas):
                    antenna_pos = antenna_positions[antenna_idx]
                    # 使用遮挡检测函数
                    is_obstructing, line_distance = check_obstruction(
                        antenna_pos, all_tag_positions[tag_idx],
                        chair_positions[chair_idx], chair_size
                    )
                    if is_obstructing:
                        has_obstruction = True
                        # 遮挡强度计算
                        blockage_strength = material_coef * (
                            1.0 - line_distance / (chair_size * 2.0)
                        )
                        obstruction_factor += blockage_strength

                # 只有当存在遮挡关系时才建立边
                if has_obstruction:
                    # 计算标签节点到椅子的欧氏距离
                    pos_dist = torch.sqrt(
                        torch.sum(
                            (all_tag_positions[tag_idx] - chair_positions[chair_idx])**2
                        )
                    )

                    # 参数设置
                    w1 = edge_attr_weights['w1']  # 距离影响权重
                    w2 = edge_attr_weights['w2']  # RSSI影响权重
                    w3 = edge_attr_weights['w3']  # 材质和大小影响权重

                    # 1. 距离项: 1/(1+d)
                    distance_factor = 1.0 / (1.0 + pos_dist.item())

                    # 2. RSSI均值项
                    # 假设RSSI值存储在features的前4个维度
                    if all_tag_features.shape[1] >= 4:
                        rssi_values = all_tag_features[tag_idx, :4]
                        rssi_mean = torch.mean(rssi_values).item()
                    else:
                        # 如果没有足够的RSSI值，使用特征的平均值
                        rssi_mean = torch.mean(all_tag_features[tag_idx]).item()

                    # 3. 材质与大小的衰减项
                    # L_chair代表材质系数，size代表椅子大小
                    L_chair = material_coef
                    if pos_dist.item() > 0:
                        decay_factor = L_chair * torch.exp(
                            torch.tensor(-chair_size / (2 * pos_dist.item()))
                        ).item()
                    else:
                        # 避免除以零
                        decay_factor = L_chair * torch.exp(
                            torch.tensor(-chair_size / 0.01)
                        ).item()

                    # 4. 多径效应影响
                    multipath_factor = 0.0
                    if pos_dist.item() > 0:
                        multipath_factor = 0.2 * L_chair * np.sin(pos_dist.item() * 5)

                    # 5. 反射与散射效应（特别是金属材质）
                    reflection_factor = 0.0
                    if L_chair > 3:  # 金属材质
                        reflection_factor = 0.3 * np.exp(-pos_dist.item() / 3)

                    # 组合所有项
                    physics_effect = (
                        w1 * distance_factor + w2 * rssi_mean + w3 * (
                            decay_factor + multipath_factor + reflection_factor +
                            1.5 * obstruction_factor
                        )
                    )

                    # 添加双向边
                    tag_to_chair_edges.append([tag_idx, chair_idx])
                    chair_to_tag_edges.append([chair_idx, tag_idx])

                    # 边属性（物理效应）
                    tag_to_chair_attrs.append(physics_effect)
                    chair_to_tag_attrs.append(physics_effect)

        # 转换为张量并设置边
        if tag_to_chair_edges:
            tag_to_chair_edge_index = torch.tensor(
                tag_to_chair_edges, dtype=torch.long
            ).t().to(device)
            chair_to_tag_edge_index = torch.tensor(
                chair_to_tag_edges, dtype=torch.long
            ).t().to(device)

            tag_to_chair_edge_attr = torch.tensor(
                tag_to_chair_attrs, dtype=torch.float32
            ).view(-1, 1).to(device)
            chair_to_tag_edge_attr = torch.tensor(
                chair_to_tag_attrs, dtype=torch.float32
            ).view(-1, 1).to(device)

            new_data['tag', 'to', 'chair'].edge_index = tag_to_chair_edge_index
            new_data['tag', 'to', 'chair'].edge_attr = tag_to_chair_edge_attr

            new_data['chair', 'to', 'tag'].edge_index = chair_to_tag_edge_index
            new_data['chair', 'to', 'tag'].edge_attr = chair_to_tag_edge_attr

        # 创建椅子-天线和天线-椅子之间的边
        chair_to_antenna_edges = []
        antenna_to_chair_edges = []
        chair_to_antenna_attrs = []
        antenna_to_chair_attrs = []

        # 为每个椅子添加到各个天线的边
        for chair_idx in range(num_chairs):
            # 获取椅子信息
            chair_size = chair_features[chair_idx, 0].item()
            material_coef = chair_features[chair_idx, 1].item()

            for antenna_idx in range(num_antennas):
                # 计算椅子到天线的欧氏距离
                pos_dist = torch.sqrt(
                    torch.sum(
                        (chair_positions[chair_idx] - antenna_positions[antenna_idx])**2
                    )
                )

                # 应用新的物理公式：椅子对天线信号的影响
                dca = pos_dist.item()
                L_chair = material_coef

                # 基础边属性计算公式: chair-to-antenna = 20log10(dca) + Lchair ⋅ (dca/size)
                if dca > 0:
                    base_attr = 20 * np.log10(dca) + L_chair * (dca / chair_size)
                else:
                    # 避免log(0)错误，使用一个很小的值
                    base_attr = 20 * np.log10(0.01) + L_chair * (0.01 / chair_size)

                # 添加多径效应影响
                multipath_factor = 0.2 * L_chair * np.sin(dca * 5)

                # 添加反射与散射效应（特别是金属材质）
                reflection_factor = 0
                if L_chair > 3:  # 金属材质
                    reflection_factor = 0.3 * np.exp(-dca / 3)

                # 检查信号遮挡效应
                obstruction_factor = 0.0
                # 检查是否有天线到标签的信号被椅子遮挡
                for tag_idx in range(len(all_tag_features)):
                    tag_pos = all_tag_positions[tag_idx]
                    # 使用遮挡检测函数
                    is_obstructing, line_distance = check_obstruction(
                        antenna_positions[antenna_idx], tag_pos,
                        chair_positions[chair_idx], chair_size
                    )
                    if is_obstructing:
                        # 遮挡强度计算
                        blockage_strength = L_chair * (
                            1.0 - line_distance / (chair_size * 2.0)
                        )
                        obstruction_factor += blockage_strength

                # 组合所有项
                physics_effect = (
                    base_attr + multipath_factor + reflection_factor +
                    1.5 * obstruction_factor
                )

                # 添加双向边
                chair_to_antenna_edges.append([chair_idx, antenna_idx])
                antenna_to_chair_edges.append([antenna_idx, chair_idx])

                # 边属性（物理效应）
                chair_to_antenna_attrs.append(physics_effect)
                antenna_to_chair_attrs.append(physics_effect)

        # 转换为张量并设置边
        if chair_to_antenna_edges:
            chair_to_antenna_edge_index = torch.tensor(
                chair_to_antenna_edges, dtype=torch.long
            ).t().to(device)
            antenna_to_chair_edge_index = torch.tensor(
                antenna_to_chair_edges, dtype=torch.long
            ).t().to(device)

            chair_to_antenna_edge_attr = torch.tensor(
                chair_to_antenna_attrs, dtype=torch.float32
            ).view(-1, 1).to(device)
            antenna_to_chair_edge_attr = torch.tensor(
                antenna_to_chair_attrs, dtype=torch.float32
            ).view(-1, 1).to(device)

            new_data['chair', 'to', 'antenna'].edge_index = chair_to_antenna_edge_index
            new_data['chair', 'to', 'antenna'].edge_attr = chair_to_antenna_edge_attr

            new_data['antenna', 'to', 'chair'].edge_index = antenna_to_chair_edge_index
            new_data['antenna', 'to', 'chair'].edge_attr = antenna_to_chair_edge_attr

            # 创建遮挡关系边：检测椅子是否遮挡天线与标签之间的信号
            chair_obstructs_tag_edges = []
            chair_obstructs_tag_attrs = []
            obstruction_counts = {}  # 记录每个标签被遮挡的次数

            # 初始化遮挡计数
            for tag_idx in range(len(all_tag_features)):
                obstruction_counts[tag_idx] = 0

            # 检查每个(椅子, 标签, 天线)组合是否存在遮挡
            for tag_idx in range(len(all_tag_features)):
                tag_pos = all_tag_positions[tag_idx]

                for chair_idx in range(num_chairs):
                    chair_pos = chair_positions[chair_idx]
                    chair_size = chair_features[chair_idx, 0].item()
                    material_coef = chair_features[chair_idx, 1].item()

                    # 检查是否存在遮挡（任何天线到该标签的路径被遮挡）
                    for antenna_idx in range(num_antennas):
                        antenna_pos = antenna_positions[antenna_idx]

                        # 使用遮挡检测函数
                        is_obstructing, line_distance = check_obstruction(
                            antenna_pos, tag_pos, chair_pos, chair_size
                        )

                        if is_obstructing:
                            # 计算遮挡强度：基于材质、距离和椅子大小
                            blockage_strength = material_coef * (
                                1.0 - line_distance / (chair_size * 2.0)
                            )

                            # 添加遮挡边
                            chair_obstructs_tag_edges.append([chair_idx, tag_idx])
                            chair_obstructs_tag_attrs.append(blockage_strength)

                            # 记录遮挡次数
                            obstruction_counts[tag_idx] += 1

                            # 找到一个遮挡就足够了，不需要继续检查其他天线
                            break

            # 设置遮挡关系边
            if chair_obstructs_tag_edges:
                # 转换为张量
                chair_obstructs_tag_edge_index = torch.tensor(
                    chair_obstructs_tag_edges, dtype=torch.long
                ).t().to(device)
                chair_obstructs_tag_edge_attr = torch.tensor(
                    chair_obstructs_tag_attrs, dtype=torch.float32
                ).view(-1, 1).to(device)

                # 添加到异构图
                new_data['chair', 'obstructs',
                         'tag'].edge_index = chair_obstructs_tag_edge_index
                new_data['chair', 'obstructs',
                         'tag'].edge_attr = chair_obstructs_tag_edge_attr

            # 为每个标签添加遮挡计数作为特征
            obstruction_feature = torch.zeros(
                len(all_tag_features), 1, dtype=torch.float32
            ).to(device)
            for tag_idx, count in obstruction_counts.items():
                obstruction_feature[tag_idx, 0] = count

            # 添加遮挡特征到标签节点
            new_data['tag'].obstruction_count = obstruction_feature

    return new_data
