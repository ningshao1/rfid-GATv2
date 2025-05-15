# -*- coding: utf-8 -*-
"""
异构图神经网络模型的训练函数
"""

import torch
import numpy as np
from models.heterogeneous.model import HeterogeneousGNNModel
from models.heterogeneous.utils import create_heterogeneous_graph_data


def train_hetero_model(
    features_norm,
    labels_norm,
    antenna_locations_norm,
    train_mask,
    val_mask,
    test_mask,
    labels_scaler,
    device,
    config,
    hidden_channels=64,
    heads=3,
    lr=0.005,
    weight_decay=5e-4,
    chair_info=None,
    edge_attr_weights=None
):
    """
    训练异构图神经网络模型

    参数:
        features_norm: 标准化的特征
        labels_norm: 标准化的标签
        antenna_locations_norm: 标准化的天线位置
        train_mask: 训练掩码
        val_mask: 验证掩码
        test_mask: 测试掩码
        labels_scaler: 标签缩放器，用于反向转换预测结果
        device: 计算设备
        config: 配置字典
        hidden_channels: 隐藏层神经元数量
        heads: 注意力头数量
        lr: 学习率
        weight_decay: 权重衰减
        chair_info: 椅子信息字典，如果不为None则添加椅子节点
        edge_attr_weights: 边属性计算的权重参数字典

    返回:
        元组 (best_val_avg_distance, best_val_loss, model, train_losses, val_losses)
    """
    # 确保每次运行结果的可重现性
    random_seed = config['RANDOM_SEED']
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 创建异构图数据
    hetero_data = create_heterogeneous_graph_data(
        features_norm,
        labels_norm,
        antenna_locations_norm,
        k=config['K'],
        device=device,
        chair_info=chair_info,
        edge_attr_weights=edge_attr_weights
    )

    # 创建异构图模型
    torch.manual_seed(config['RANDOM_SEED'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['RANDOM_SEED'])
        torch.cuda.manual_seed_all(config['RANDOM_SEED'])
    hetero_model = HeterogeneousGNNModel(
        in_channels=features_norm.shape[1],
        hidden_channels=hidden_channels,
        out_channels=2,
        heads=heads
    ).to(device)

    # 添加训练和验证掩码到异构数据
    hetero_data['tag'].train_mask = train_mask | test_mask
    hetero_data['tag'].val_mask = val_mask

    # 创建优化器
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    optimizer = torch.optim.Adam(
        hetero_model.parameters(), lr=lr, weight_decay=weight_decay
    )
    loss_fn = torch.nn.MSELoss()

    # 学习率调整器 - 降低调整频率以增加稳定性
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=30, min_lr=1e-6, verbose=False
    )

    # 为MinMaxScaler参数创建张量
    data_min = torch.as_tensor(labels_scaler.data_min_, dtype=torch.float32).to(device)
    data_range = torch.as_tensor(labels_scaler.data_range_,
                                 dtype=torch.float32).to(device)

    # 开始训练
    best_val_loss = float('inf')
    best_model = None
    patience = config.get('PATIENCE', 70)  # 增加早停耐心值
    counter = 0  # 计数器
    best_val_avg_distance = float('inf')

    # 使用计数器记录连续验证集指标没有改善的次数
    no_improve_counter = 0
    min_epochs = 200  # 增加最小训练轮次
    best_epoch = 0  # 记录最佳模型的轮次

    # 初始化损失记录
    train_losses = []
    val_losses = []

    # 准备edge_index和edge_attr字典
    edge_index_dict = {
        ('tag', 'to', 'tag'): hetero_data['tag', 'to', 'tag'].edge_index,
        ('tag', 'to', 'antenna'): hetero_data['tag', 'to', 'antenna'].edge_index,
        ('antenna', 'to', 'tag'): hetero_data['antenna', 'to', 'tag'].edge_index
    }

    edge_attr_dict = {
        ('tag', 'to', 'tag'): hetero_data['tag', 'to', 'tag'].edge_attr,
        ('tag', 'to', 'antenna'): hetero_data['tag', 'to', 'antenna'].edge_attr,
        ('antenna', 'to', 'tag'): hetero_data['antenna', 'to', 'tag'].edge_attr
    }

    # 如果存在椅子节点，添加相关的边
    if 'chair' in hetero_data:
        # 椅子-标签边
        if ('tag', 'to', 'chair') in hetero_data:
            edge_index_dict[('tag', 'to', 'chair')] = hetero_data['tag', 'to',
                                                                  'chair'].edge_index
            edge_attr_dict[('tag', 'to', 'chair')] = hetero_data['tag', 'to',
                                                                 'chair'].edge_attr

        if ('chair', 'to', 'tag') in hetero_data:
            edge_index_dict[('chair', 'to', 'tag')] = hetero_data['chair', 'to',
                                                                  'tag'].edge_index
            edge_attr_dict[('chair', 'to', 'tag')] = hetero_data['chair', 'to',
                                                                 'tag'].edge_attr

        # 椅子-天线边
        if ('chair', 'to', 'antenna') in hetero_data:
            edge_index_dict[('chair', 'to', 'antenna')
                           ] = hetero_data['chair', 'to', 'antenna'].edge_index
            edge_attr_dict[('chair', 'to', 'antenna')
                          ] = hetero_data['chair', 'to', 'antenna'].edge_attr

        if ('antenna', 'to', 'chair') in hetero_data:
            edge_index_dict[('antenna', 'to', 'chair')
                           ] = hetero_data['antenna', 'to', 'chair'].edge_index
            edge_attr_dict[('antenna', 'to', 'chair')] = hetero_data['antenna', 'to',
                                                                     'chair'].edge_attr

    # 准备节点特征字典
    x_dict = {'tag': hetero_data['tag'].x, 'antenna': hetero_data['antenna'].x}

    # 如果存在椅子节点，添加到特征字典
    if 'chair' in hetero_data:
        x_dict['chair'] = hetero_data['chair'].x

    # 训练循环
    for epoch in range(config.get('EPOCHS', 1000)):
        # 每个epoch重设随机种子以增加稳定性
        # epoch_seed = random_seed + epoch
        # np.random.seed(epoch_seed)
        # torch.manual_seed(epoch_seed)
        # if torch.cuda.is_available():
        #     torch.cuda.manual_seed(epoch_seed)
        #     torch.cuda.manual_seed_all(epoch_seed)

        # 训练阶段
        hetero_model.train()
        optimizer.zero_grad()

        # 前向传播
        out = hetero_model(x_dict, edge_index_dict, edge_attr_dict)

        # 计算损失 - 只考虑训练掩码中的标签节点
        train_loss = loss_fn(
            out[hetero_data['tag'].train_mask],
            hetero_data['tag'].y[hetero_data['tag'].train_mask]
        )
        train_loss.backward()
        optimizer.step()

        # 验证阶段
        hetero_model.eval()
        with torch.no_grad():
            # 前向传播
            out = hetero_model(x_dict, edge_index_dict, edge_attr_dict)

            # 计算验证损失 - 只考虑验证掩码中的标签节点
            val_loss = loss_fn(
                out[hetero_data['tag'].val_mask],
                hetero_data['tag'].y[hetero_data['tag'].val_mask]
            )

            # 将预测结果转换回原始比例（逆MinMaxScaler）
            out_orig = out * data_range + data_min
            y_orig = hetero_data['tag'].y * data_range + data_min

            # 计算训练集和验证集的距离误差
            train_distances = torch.sqrt(
                torch.sum((
                    out_orig[hetero_data['tag'].train_mask] -
                    y_orig[hetero_data['tag'].train_mask]
                )**2,
                          dim=1)
            )
            train_accuracy = (train_distances < 0.3).float().mean().item() * 100
            train_avg_distance = train_distances.mean().item()

            val_distances = torch.sqrt(
                torch.sum((
                    out_orig[hetero_data['tag'].val_mask] -
                    y_orig[hetero_data['tag'].val_mask]
                )**2,
                          dim=1)
            )
            val_accuracy = (val_distances < 0.3).float().mean().item() * 100
            val_avg_distance = val_distances.mean().item()

        # 调整学习率
        scheduler.step(val_loss)

        # 保存每个epoch的损失值
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())

        # 早停检查 - 同时考虑验证损失和平均距离
        improved = False

        # 修改优化条件，平均距离的改善需要达到更高阈值才算提升
        if val_avg_distance < best_val_avg_distance * 0.99:  # 至少改善1%
            best_val_avg_distance = val_avg_distance
            best_val_loss = val_loss
            best_model = hetero_model.state_dict().copy()
            best_epoch = epoch
            improved = True
            no_improve_counter = 0
        else:
            no_improve_counter += 1

            # 如果距离没有改善但损失有显著改善也算提升
            if val_loss < best_val_loss * 0.95:  # 损失降低了至少5%
                best_val_loss = val_loss
                best_model = hetero_model.state_dict().copy()
                best_epoch = epoch
                improved = True
                no_improve_counter = 0

        # 只有在超过最小训练轮次后才考虑早停
        if no_improve_counter >= patience and epoch >= min_epochs:
            if config.get('TRAIN_LOG', False):
                # 使用RFIDLocalization实例通过传入的config获取
                rfid_instance = config.get('RFID_INSTANCE', None)
                log_message = (
                    f"异构图模型轮次 {epoch}\n"
                    f"训练集 - 损失: {train_loss.item():.4f}, 准确率: {train_accuracy:.2f}%, 平均误差: {train_avg_distance:.2f}米\n"
                    f"验证集 - 损失: {val_loss.item():.4f}, 准确率: {val_accuracy:.2f}%, 平均误差: {val_avg_distance:.2f}米\n"
                    f"\n触发早停！在轮次 {epoch} 停止训练\n"
                    f"最佳验证损失: {best_val_loss:.4f}, 最佳验证集平均误差: {best_val_avg_distance:.2f}米, 最佳轮次: {best_epoch}"
                )

                if rfid_instance and hasattr(rfid_instance, 'train_logger'):
                    rfid_instance.train_logger.info(log_message)

            # 加载最佳模型
            hetero_model.load_state_dict(best_model)
            break

        if epoch % 100 == 0 and config.get('TRAIN_LOG', False):
            rfid_instance = config.get('RFID_INSTANCE', None)
            current_lr = optimizer.param_groups[0]['lr']
            log_message = (
                f"异构图模型轮次 {epoch}\n"
                f"训练集 - 损失: {train_loss.item():.4f}, 准确率: {train_accuracy:.2f}%, 平均误差: {train_avg_distance:.2f}米\n"
                f"验证集 - 损失: {val_loss.item():.4f}, 准确率: {val_accuracy:.2f}%, 平均误差: {val_avg_distance:.2f}米\n"
                f"当前学习率: {current_lr:.6f}"
            )

            if rfid_instance and hasattr(rfid_instance, 'train_logger'):
                rfid_instance.train_logger.info(log_message)

    return best_val_avg_distance, best_val_loss.item(
    ), hetero_model, train_losses, val_losses
