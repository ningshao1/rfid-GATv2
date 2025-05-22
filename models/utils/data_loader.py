# -*- coding: utf-8 -*-
"""
数据加载和预处理工具函数
"""

import numpy as np
import pandas as pd
import torch


def load_and_preprocess_test_data(test_csv_path, scaler_rssi=None, scaler_phase=None):
    """
    加载并标准化测试集数据。

    参数:
        test_csv_path: 测试集CSV文件路径
        scaler_rssi: RSSI值的标准化器
        scaler_phase: 相位值的标准化器

    返回：
        test_features: 原始特征张量
        test_labels: 原始标签张量
        test_features_np: numpy格式的原始特征
        test_labels_np: numpy格式的原始标签
        test_features_norm: 标准化后的特征（如果提供了标准化器）
    """
    df_test = pd.read_csv(test_csv_path)
    test_features = torch.tensor(
        df_test[[
            'rssi_antenna1', 'rssi_antenna2', 'rssi_antenna3', 'rssi_antenna4',
            "wrapped_phase_antenna1", "wrapped_phase_antenna2",
            "wrapped_phase_antenna3", "wrapped_phase_antenna4"
        ]].values,
        dtype=torch.float32
    )
    # test_features[:, 4:8] = 0
    test_labels = torch.tensor(
        df_test[['true_x', 'true_y']].values, dtype=torch.float32
    )
    test_features_np = test_features.cpu().numpy()
    test_labels_np = test_labels.cpu().numpy()
    if scaler_rssi is not None and scaler_phase is not None:
        rssi_norm = scaler_rssi.transform(test_features_np[:, :4])
        phase_norm = scaler_phase.transform(test_features_np[:, 4:8])
        test_features_norm = np.hstack([rssi_norm, phase_norm])
    else:
        test_features_norm = None
    return test_features, test_labels, test_features_np, test_labels_np, test_features_norm
