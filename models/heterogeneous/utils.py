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


# 批量检查遮挡关系
def batch_check_obstruction(
    antenna_positions, tag_positions, chair_positions, chair_sizes
):
    """
    批量检测椅子是否位于天线和标签之间，形成遮挡

    参数:
        antenna_positions: 所有天线位置坐标 [num_antennas, 2]
        tag_positions: 所有标签位置坐标 [num_tags, 2]
        chair_positions: 所有椅子位置坐标 [num_chairs, 2]
        chair_sizes: 所有椅子大小 [num_chairs]

    返回:
        obstruction_matrix: 形状为 [num_chairs, num_tags, num_antennas] 的布尔矩阵，
                           表示每个(椅子,标签,天线)组合是否存在遮挡
        distance_matrix: 形状为 [num_chairs, num_tags, num_antennas] 的距离矩阵
    """
    # 转换为numpy数组
    if isinstance(antenna_positions, torch.Tensor):
        antenna_positions = antenna_positions.cpu().numpy()
    if isinstance(tag_positions, torch.Tensor):
        tag_positions = tag_positions.cpu().numpy()
    if isinstance(chair_positions, torch.Tensor):
        chair_positions = chair_positions.cpu().numpy()
    if isinstance(chair_sizes, torch.Tensor):
        chair_sizes = chair_sizes.cpu().numpy()

    num_antennas = len(antenna_positions)
    num_tags = len(tag_positions)
    num_chairs = len(chair_positions)

    # 初始化结果矩阵
    obstruction_matrix = np.zeros((num_chairs, num_tags, num_antennas), dtype=bool)
    distance_matrix = np.zeros((num_chairs, num_tags, num_antennas))

    # 计算所有组合的遮挡关系
    for chair_idx in range(num_chairs):
        chair_pos = chair_positions[chair_idx]
        chair_size = chair_sizes[chair_idx]

        for tag_idx in range(num_tags):
            tag_pos = tag_positions[tag_idx]

            for antenna_idx in range(num_antennas):
                antenna_pos = antenna_positions[antenna_idx]

                is_obstructing, line_distance = check_obstruction(
                    antenna_pos, tag_pos, chair_pos, chair_size
                )

                obstruction_matrix[chair_idx, tag_idx, antenna_idx] = is_obstructing
                distance_matrix[chair_idx, tag_idx, antenna_idx] = line_distance

    return obstruction_matrix, distance_matrix


def create_hetero_graph_edges(
    data,
    tag_features,
    tag_positions,
    antenna_positions,
    chair_info=None,
    k=7,
    device='cpu',
    edge_attr_weights=None
):
    """
    为异构图创建各种类型的边

    参数:
        data: HeteroData对象
        tag_features: 标签节点特征
        tag_positions: 标签节点位置
        antenna_positions: 天线节点位置
        chair_info: 椅子信息，包含椅子特征和位置（可选）
        k: KNN的K值
        device: 计算设备
        edge_attr_weights: 边属性计算的权重参数字典

    返回:
        更新后的HeteroData对象
    """

    # 获取节点数量
    num_tags = len(tag_features)
    num_antennas = len(antenna_positions)

    # 使用特征和位置的组合来计算KNN
    combined_features = torch.cat(
        [
            0.95 * tag_features,  # 给特征较小的权重
            0.05 * tag_positions  # 给位置较大的权重
        ],
        dim=1
    ).cpu().numpy()

    # 计算KNN邻接矩阵
    adj_matrix = kneighbors_graph(
        combined_features,
        n_neighbors=min(k, num_tags - 1) if num_tags > 1 else 1,
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

    # 使用矩阵操作批量计算距离
    tag_positions_expanded = tag_positions.unsqueeze(1)  # [num_tags, 1, 2]
    antenna_positions_expanded = antenna_positions.unsqueeze(0)  # [1, num_antennas, 2]

    # 计算欧氏距离矩阵 [num_tags, num_antennas]
    distances = torch.sqrt(
        torch.sum((tag_positions_expanded - antenna_positions_expanded)**2, dim=2)
    )

    # 预先计算标签节点特征范数
    tag_features_norm = torch.norm(tag_features, dim=1).unsqueeze(1)  # [num_tags, 1]

    # 特征距离
    feat_weight = 0.2  # 特征权重
    dist_matrix = (1 - feat_weight) * distances + feat_weight * tag_features_norm

    # 创建边索引和属性
    tag_to_antenna_edges = []
    antenna_to_tag_edges = []
    tag_to_antenna_attrs = []
    antenna_to_tag_attrs = []

    for tag_idx in range(num_tags):
        for antenna_idx in range(num_antennas):
            # 添加双向边
            tag_to_antenna_edges.append([tag_idx, antenna_idx])
            antenna_to_tag_edges.append([antenna_idx, tag_idx])

            # 边属性（距离）
            dist = dist_matrix[tag_idx, antenna_idx].item()
            tag_to_antenna_attrs.append(dist)
            antenna_to_tag_attrs.append(dist)

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

    # 3. 如果存在椅子节点，创建椅子相关的边
    if chair_info is not None and hasattr(data,
                                          'chair') and hasattr(data['chair'], 'x'):

        # 获取椅子节点
        chair_features = data['chair'].x
        chair_positions = data['chair'].y

        num_chairs = len(chair_positions)

        # 提取椅子大小和材质系数
        chair_sizes = chair_features[:, 0]  # 第一列是椅子大小
        material_coefs = chair_features[:, 1]  # 第二列是材质系数

        # 批量计算tag到chair的距离矩阵 [num_tags, num_chairs]
        tag_positions_expanded = tag_positions.unsqueeze(1)  # [num_tags, 1, 2]
        chair_positions_expanded = chair_positions.unsqueeze(0)  # [1, num_chairs, 2]
        tag_chair_distances = torch.sqrt(
            torch.sum((tag_positions_expanded - chair_positions_expanded)**2, dim=2)
        )

        # 预先计算tag的RSSI均值 [num_tags]
        # 假设tag_features前4个维度是RSSI
        if tag_features.shape[1] >= 4:
            rssi_features = tag_features[:, :4]
            tag_rssi_mean = torch.mean(rssi_features,
                                       dim=1).unsqueeze(1)  # [num_tags, 1]
        else:
            # 如果没有足够的维度，使用标准化后的特征代替
            tag_rssi_mean = torch.norm(tag_features,
                                       dim=1).unsqueeze(1)  # [num_tags, 1]

        # 批量计算遮挡关系
        obstruction_matrix, distance_matrix = batch_check_obstruction(
            antenna_positions.cpu().numpy(),
            tag_positions.cpu().numpy(),
            chair_positions.cpu().numpy(),
            chair_sizes.cpu().numpy()
        )

        # 创建边容器
        tag_to_chair_edges = []
        chair_to_tag_edges = []
        tag_to_chair_attrs = []
        chair_to_tag_attrs = []

        # 设置权重参数
        if edge_attr_weights is None:
            edge_attr_weights = {'w1': 0.6, 'w2': 0.1, 'w3': 0.3}
        w1 = edge_attr_weights.get('w1', 0.6)  # 距离影响权重
        w2 = edge_attr_weights.get('w2', 0.1)  # RSSI影响权重
        w3 = edge_attr_weights.get('w3', 0.3)  # 材质和大小影响权重

        # 计算边属性
        for chair_idx in range(num_chairs):
            chair_size = chair_sizes[chair_idx].item()
            material_coef = material_coefs[chair_idx].item()

            # 对每个可能的标签创建边
            for tag_idx in range(num_tags):
                distance = tag_chair_distances[tag_idx, chair_idx].item()
                rssi_mean = tag_rssi_mean[tag_idx].item()

                # 使用距离计算物理效应
                # 距离越远，影响越小
                distance_factor = 1.0 / (1.0 + distance)

                # 物理效应计算
                # 椅子尺寸对信号的衰减和多径
                decay_factor = chair_size * 0.5 if chair_size > 0.5 else 0.1
                # 多径效应，取决于椅子大小和材质
                multipath_factor = (chair_size * material_coef) * 0.3
                # 反射效应，主要取决于材质
                reflection_factor = material_coef * 0.4 if material_coef > 3 else 0.0

                # 遮挡因子
                obstruction_factor = 0.0
                for antenna_idx in range(num_antennas):
                    if obstruction_matrix[chair_idx, tag_idx, antenna_idx]:
                        # 粗略估计遮挡强度
                        obstruction_factor += material_coef * 0.5
                        break

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

                # 边属性
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

        # 预先计算所有椅子到所有天线的距离矩阵
        chair_positions_expanded = chair_positions.unsqueeze(1)  # [num_chairs, 1, 2]
        antenna_positions_expanded = antenna_positions.unsqueeze(
            0
        )  # [1, num_antennas, 2]

        # 计算欧氏距离矩阵 [num_chairs, num_antennas]
        chair_antenna_distances = torch.sqrt(
            torch.sum((chair_positions_expanded - antenna_positions_expanded)**2, dim=2)
        )

        # 创建边容器
        chair_to_antenna_edges = []
        antenna_to_chair_edges = []
        chair_to_antenna_attrs = []
        antenna_to_chair_attrs = []

        # 遍历所有可能的椅子和天线对
        for chair_idx in range(num_chairs):
            chair_size = chair_sizes[chair_idx].item()
            material_coef = material_coefs[chair_idx].item()

            for antenna_idx in range(num_antennas):
                distance = chair_antenna_distances[chair_idx, antenna_idx].item()

                # 使用距离计算物理效应
                # 距离越远，影响越小
                distance_factor = 1.0 / (1.0 + distance)

                # 信号衰减，与椅子尺寸和材质有关
                decay_factor = chair_size * 0.5 * material_coef

                # 反射效应，主要取决于材质
                reflection_factor = material_coef * 0.4 if material_coef > 1.5 else 0.1

                # 组合所有项
                physics_effect = w1 * distance_factor + w3 * (
                    decay_factor + reflection_factor
                )

                # 添加双向边
                chair_to_antenna_edges.append([chair_idx, antenna_idx])
                antenna_to_chair_edges.append([antenna_idx, chair_idx])

                # 边属性
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

        chair_obstructs_tag_edges = []
        chair_obstructs_tag_attrs = []

        # 直接使用计算得到的遮挡矩阵创建边
        for chair_idx in range(num_chairs):
            chair_size = chair_sizes[chair_idx].item()
            material_coef = material_coefs[chair_idx].item()

            for tag_idx in range(num_tags):
                if np.any(obstruction_matrix[chair_idx, tag_idx]):
                    # 找到遮挡的天线
                    for antenna_idx in range(num_antennas):
                        if obstruction_matrix[chair_idx, tag_idx, antenna_idx]:
                            blockage_strength = material_coef * 0.5

                            chair_obstructs_tag_edges.append([chair_idx, tag_idx])
                            chair_obstructs_tag_attrs.append(blockage_strength)
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
            data['chair', 'obstructs', 'tag'].edge_attr = chair_obstructs_tag_edge_attr

        # 3.4 为每个标签添加遮挡计数作为特征
        obstruction_counts = torch.zeros(num_tags, dtype=torch.float32, device=device)
        for tag_idx in range(num_tags):
            # 统计每个标签被遮挡的情况
            obstruction_count = 0
            for chair_idx in range(num_chairs):
                if np.any(obstruction_matrix[chair_idx, tag_idx]):
                    obstruction_count += 1
            obstruction_counts[tag_idx] = obstruction_count

        # 添加到数据中
        if hasattr(data, 'tag'):
            data['tag'].obstruction_count = obstruction_counts.float().view(-1, 1
                                                                           ).to(device)

    return data


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

    # 添加椅子节点（如果提供了椅子信息）
    if chair_info is not None and len(chair_info) > 0:
        # 确保chair_info是列表类型
        if not isinstance(chair_info, list):
            chair_info = [chair_info]

        num_chairs = len(chair_info)

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

    # 创建异构图的边
    data = create_hetero_graph_edges(
        data, features_norm, labels_norm, antenna_positions,
        chair_info if chair_info and len(chair_info) > 0 else None, k, device,
        edge_attr_weights
    )

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
    chair_info = None
    try:
        # 尝试访问椅子节点特征
        chair_features = hetero_data['chair'].x
        chair_positions = hetero_data['chair'].y
        # 设置椅子节点特征和位置
        new_data['chair'].x = chair_features
        new_data['chair'].y = chair_positions
        chair_info = True
    except (KeyError, AttributeError):
        # 如果发生KeyError或AttributeError，说明没有椅子节点
        pass

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

    # 创建异构图的边
    new_data = create_hetero_graph_edges(
        new_data, all_tag_features, all_tag_positions, antenna_positions, chair_info, k,
        device, edge_attr_weights
    )

    return new_data
