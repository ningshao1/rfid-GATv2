# -*- coding: utf-8 -*-
"""
数据增强工具函数
"""

import torch
import numpy as np


def add_gaussian_noise(X, noise_scale=0.15):
    """
    添加高斯噪声到数据中

    参数:
        X: 输入数据张量
        noise_scale: 噪声比例

    返回:
        添加噪声后的数据张量
    """
    return torch.tensor(
        X.cpu().numpy() + np.random.normal(0, noise_scale,
                                           X.cpu().numpy().shape),
        dtype=torch.float32
    )


def scale_features(X, scale_range=(0.9, 1.1)):
    """
    对特征进行缩放

    参数:
        X: 输入数据张量
        scale_range: 缩放范围元组 (min_scale, max_scale)

    返回:
        缩放后的数据张量
    """
    scale_factors = np.random.uniform(
        scale_range[0], scale_range[1], (X.cpu().numpy().shape[0], 1)
    )
    return torch.tensor(X.cpu().numpy() * scale_factors, dtype=torch.float32)


def mix_rssi_phase(X, rssi_range=(-0.03, 0.03), phase_range=(-0.03, 0.03)):
    """
    对RSSI和相位值进行混合扰动

    参数:
        X: 输入数据张量
        rssi_range: RSSI扰动范围
        phase_range: 相位扰动范围

    返回:
        扰动后的数据张量
    """
    X_mixed = X.cpu().numpy().copy()
    X_mixed[:, :4] += np.random.uniform(
        rssi_range[0], rssi_range[1], (X.cpu().numpy().shape[0], 4)
    )
    X_mixed[:, 4:] += np.random.uniform(
        phase_range[0], phase_range[1], (X.cpu().numpy().shape[0], 4)
    )
    return torch.tensor(X_mixed, dtype=torch.float32)


def apply_data_augmentation(X, y, device):
    """
    应用所有数据增强方法并合并结果

    参数:
        X: 输入特征张量
        y: 标签张量
        device: 计算设备

    返回:
        增强后的特征和标签张量
    """
    # 添加高斯噪声
    X_noisy = add_gaussian_noise(X).to(device)

    # 特征缩放
    X_scaled = scale_features(X).to(device)

    # RSSI和相位混合扰动
    X_mixed = mix_rssi_phase(X).to(device)

    # 合并所有增强数据
    X_augmented = torch.cat([X, X_noisy, X_scaled, X_mixed], dim=0)
    y_augmented = torch.cat([y] * 4, dim=0)

    return X_augmented, y_augmented
