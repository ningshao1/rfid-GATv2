# -*- coding: utf-8 -*-
"""
MLP模型的工具函数
"""

import torch
import numpy as np
from models.utils.data_augmentation import apply_data_augmentation
from models.gat.utils import create_data_masks


def train_mlp_model(
    localization, hidden_channels=128, dropout=0.1, lr=0.001, weight_decay=0.0005
):
    """
    训练MLP模型，支持超参数传递，返回验证集损失、平均误差和最佳模型参数

    参数:
        localization: RFIDLocalization实例
        hidden_channels: 隐藏层神经元数量
        dropout: Dropout比率
        lr: 学习率
        weight_decay: 权重衰减

    返回:
        best_val_loss: 最佳验证损失
        best_val_avg_distance: 最佳验证集平均误差
        best_model: 最佳模型参数
    """
    # 创建训练、验证和测试掩码
    train_mask, val_mask, test_mask = create_data_masks(
        len(localization.features_norm), localization.config, localization.device
    )
    train_mask = train_mask | test_mask

    # 准备数据
    X = localization.features_norm.to(localization.device)
    y = localization.labels_norm.to(localization.device)

    # 数据增强
    if localization.config.get('DATA_AUGMENTATION', False):
        X_train_tensor, y_train_tensor = apply_data_augmentation(
            X[train_mask], y[train_mask], localization.device
        )
        if localization.config['TRAIN_LOG']:
            print(f"使用数据增强: 原始样本数 {len(X[train_mask])}, 增强后样本数 {len(X_train_tensor)}")
    else:
        X_train_tensor = X[train_mask]
        y_train_tensor = y[train_mask]

    # 创建MLP模型
    torch.manual_seed(localization.config['RANDOM_SEED'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(localization.config['RANDOM_SEED'])
        torch.cuda.manual_seed_all(localization.config['RANDOM_SEED'])
    from models.mlp.model import MLPLocalizationModel
    localization.mlp_model = MLPLocalizationModel(
        in_channels=X.shape[1],
        hidden_channels=hidden_channels,
        out_channels=2,
        dropout=dropout
    ).to(localization.device)

    # 为MinMaxScaler参数创建张量
    data_min = torch.as_tensor(
        localization.labels_scaler.data_min_, dtype=torch.float32
    ).to(localization.device)
    data_range = torch.as_tensor(
        localization.labels_scaler.data_range_, dtype=torch.float32
    ).to(localization.device)

    # 使用固定种子初始化优化器
    torch.manual_seed(localization.config['RANDOM_SEED'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(localization.config['RANDOM_SEED'])
        torch.cuda.manual_seed_all(localization.config['RANDOM_SEED'])
    optimizer = torch.optim.Adam(
        localization.mlp_model.parameters(), lr=lr, weight_decay=weight_decay
    )
    loss_fn = torch.nn.MSELoss()

    best_val_loss = float('inf')
    best_model = None
    patience = localization.config.get('PATIENCE', 50)  # 早停耐心值
    counter = 0  # 计数器
    best_val_avg_distance = float('inf')

    # 清空损失记录
    localization.mlp_train_losses = []
    localization.mlp_val_losses = []

    for epoch in range(localization.config.get('EPOCHS', 1000)):
        # 训练阶段
        localization.mlp_model.train()
        optimizer.zero_grad()
        out = localization.mlp_model(X_train_tensor)
        train_loss = loss_fn(out, y_train_tensor)
        train_loss.backward()
        optimizer.step()

        # 验证阶段
        localization.mlp_model.eval()
        with torch.no_grad():
            val_out = localization.mlp_model(X[val_mask])
            val_loss = loss_fn(val_out, y[val_mask])
            out_orig = val_out * data_range + data_min
            y_orig = y[val_mask] * data_range + data_min
            val_distances = torch.sqrt(torch.sum((out_orig - y_orig)**2, dim=1))
            val_accuracy = (val_distances < 0.3).float().mean().item() * 100
            val_avg_distance = val_distances.mean().item()

        # 保存每个epoch的损失值
        localization.mlp_train_losses.append(train_loss.item())
        localization.mlp_val_losses.append(val_loss.item())

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_avg_distance = val_avg_distance
            best_model = localization.mlp_model.state_dict().copy()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                if localization.config['TRAIN_LOG']:
                    print(
                        f"MLP轮次 {epoch}\n"
                        f"验证集 - 损失: {val_loss.item():.4f}, 准确率: {val_accuracy:.2f}%, 平均误差: {val_avg_distance:.2f}米"
                    )
                    print(f"\n触发早停！在轮次 {epoch} 停止训练")
                    print(f"最佳验证损失: {best_val_loss:.4f}")
                localization.mlp_model.load_state_dict(best_model)
                break

        if epoch % 100 == 0 and localization.config['TRAIN_LOG']:
            print(
                f"MLP轮次 {epoch}\n"
                f"训练集 - 损失: {train_loss.item():.4f}\n"
                f"验证集 - 损失: {val_loss.item():.4f}, 准确率: {val_accuracy:.2f}%, 平均误差: {val_avg_distance:.2f}米"
            )

    # 训练结束后加载最佳模型
    localization.mlp_model.load_state_dict(best_model)
    return best_val_loss, best_val_avg_distance, best_model


def evaluate_mlp_on_new_data(localization, test_features, test_labels):
    """
    评估MLP模型在新数据上的性能

    参数:
        localization: RFIDLocalization实例
        test_features: 测试特征
        test_labels: 测试标签

    返回:
        avg_distance: 平均预测误差
    """
    if localization.mlp_model is None:
        raise ValueError("MLP模型未训练。请先调用train_mlp_model。")

    # 获取RSSI和相位值
    rssi_values = test_features[:, :4]
    phase_values = test_features[:, 4:8]

    # 标准化特征
    rssi_norm = localization.scaler_rssi.transform(rssi_values)
    phase_norm = localization.scaler_phase.transform(phase_values)

    # 组合特征
    features_norm = np.hstack([rssi_norm, phase_norm])
    features_tensor = torch.tensor(features_norm,
                                   dtype=torch.float32).to(localization.device)

    # 预测
    localization.mlp_model.eval()
    with torch.no_grad():
        predictions = localization.mlp_model(features_tensor)

        # 反标准化以获得实际坐标
        predictions_orig = localization.labels_scaler.inverse_transform(
            predictions.cpu().numpy()
        )

        # 计算欧几里得距离误差
        distances = np.sqrt(np.sum((test_labels - predictions_orig)**2, axis=1))
        avg_distance = np.mean(distances)

    if localization.config['PREDICTION_LOG']:
        print("\nMLP模型新标签位置预测评估:")
        print(f"测试样本数量: {len(test_features)}")
        print(f"平均预测误差: {avg_distance:.2f}米")
        print(f"最大误差: {np.max(distances):.2f}米")
        print(f"最小误差: {np.min(distances):.2f}米")
        print(f"误差标准差: {np.std(distances):.2f}米")

        # 计算不同误差阈值下的准确率
        for threshold in [0.5, 1.0, 1.5, 2.0]:
            accuracy = np.mean(distances < threshold) * 100
            print(f"误差 < {threshold}米的准确率: {accuracy:.2f}%")

    return avg_distance
