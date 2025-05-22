# -*- coding: utf-8 -*-
"""
LANDMARC定位算法实现
"""

import numpy as np
import torch


def landmarc_localization(
    reference_features, reference_locations, test_features, test_labels=None, k=7
):
    """
    使用LANDMARC算法进行定位

    参数:
        reference_features: 参考标签的特征（RSSI和相位值）
        reference_locations: 参考标签的真实位置
        test_features: 测试标签的特征（RSSI和相位值）
        test_labels: 测试标签的真实位置（可选，用于评估）
        k: k近邻数量

    返回:
        预测位置和平均误差（如果提供真实位置）
    """
    reference_features = reference_features[:, :4]
    test_features = test_features[:, :4]
    # 确保数据是numpy数组格式
    if isinstance(reference_features, torch.Tensor):
        reference_features = reference_features.cpu().numpy()
    if isinstance(reference_locations, torch.Tensor):
        reference_locations = reference_locations.cpu().numpy()
    if isinstance(test_features, torch.Tensor):
        test_features = test_features.cpu().numpy()
    if test_labels is not None and isinstance(test_labels, torch.Tensor):
        test_labels = test_labels.cpu().numpy()

    # 计算测试标签预测位置
    predictions = []

    # 遍历每个测试样本
    for i in range(len(test_features)):
        # 提取当前测试标签的特征值
        test_feature = test_features[i].reshape(1, -1)

        # 计算欧氏距离（信号空间距离）
        signal_distances = np.sqrt(
            np.sum((reference_features - test_feature)**2, axis=1)
        )

        # 找到k个最近的参考标签索引
        nearest_indices = np.argsort(signal_distances)[:k]

        # 提取距离最近的k个参考标签的位置和距离
        nearest_locations = reference_locations[nearest_indices]
        nearest_distances = signal_distances[nearest_indices]

        # 计算权重 (使用距离的倒数作为权重)
        # 添加小值避免除以零
        weights = 1.0 / (nearest_distances + 1e-6)

        # 归一化权重
        weights = weights / np.sum(weights)

        # 计算加权平均位置
        predicted_location = np.sum(nearest_locations * weights.reshape(-1, 1), axis=0)
        predictions.append(predicted_location)

    predictions = np.array(predictions)

    # 如果提供了真实位置，计算误差
    avg_error = None
    if test_labels is not None:
        distances = np.sqrt(np.sum((predictions - test_labels)**2, axis=1))
        avg_error = np.mean(distances)

    return predictions, avg_error


def evaluate_landmarc(
    reference_features,
    reference_locations,
    test_features,
    test_labels,
    k=7,
    verbose=False
):
    """
    评估LANDMARC算法在测试数据上的性能

    参数:
        reference_features: 参考标签的特征
        reference_locations: 参考标签的位置
        test_features: 测试标签的特征
        test_labels: 测试标签的真实位置
        k: k近邻数量
        verbose: 是否打印详细评估信息

    返回:
        平均误差距离
    """
    predictions, avg_error = landmarc_localization(
        reference_features, reference_locations, test_features, test_labels, k
    )

    if verbose:
        # 计算各种误差指标
        distances = np.sqrt(np.sum((predictions - test_labels)**2, axis=1))

        print("\nLANDMARC算法位置预测评估:")
        print(f"测试样本数量: {len(test_features)}")
        print(f"平均预测误差: {avg_error:.2f}米")
        print(f"最大误差: {np.max(distances):.2f}米")
        print(f"最小误差: {np.min(distances):.2f}米")
        print(f"误差标准差: {np.std(distances):.2f}米")

        # 计算不同误差阈值下的准确率
        for threshold in [0.5, 1.0, 1.5, 2.0]:
            accuracy = np.mean(distances < threshold) * 100
            print(f"误差 < {threshold}米的准确率: {accuracy:.2f}%")

    return avg_error
