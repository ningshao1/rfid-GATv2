# -*- coding: utf-8 -*-
"""
KNN定位模型
"""

import numpy as np
import torch
from sklearn.neighbors import KNeighborsRegressor


def knn_localization(features, labels, n_neighbors=7):
    """
    使用KNN进行位置预测

    参数:
        features: 特征数据
        labels: 标签数据
        n_neighbors: K近邻数量

    返回:
        训练好的KNN模型
    """
    # 确保数据是numpy数组格式
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # 创建并训练KNN模型
    knn = KNeighborsRegressor(n_neighbors=n_neighbors, algorithm='auto')
    knn.fit(features, labels)

    return knn


def evaluate_knn_on_test_set(knn_model, test_features, test_labels):
    """
    在测试集上评估KNN模型

    参数:
        knn_model: 训练好的KNN模型
        test_features: 测试特征
        test_labels: 测试标签

    返回:
        平均误差距离
    """
    # 确保数据是numpy数组格式
    if isinstance(test_features, torch.Tensor):
        test_features = test_features.cpu().numpy()
    if isinstance(test_labels, torch.Tensor):
        test_labels = test_labels.cpu().numpy()

    # 预测测试集
    y_pred = knn_model.predict(test_features)

    # 计算实际米数的误差
    distances = np.sqrt(np.sum((test_labels - y_pred)**2, axis=1))
    avg_distance = np.mean(distances)

    return avg_distance
