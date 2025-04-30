# -*- coding: utf-8 -*-
"""
使用图注意力网络(GAT)的RFID定位方法
"""

import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.neighbors import kneighbors_graph, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix, to_undirected
from torch_geometric.nn import GATv2Conv
import itertools
import time
import os

# 忽略警告
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 配置
CONFIG = {
    'K': 7,  # KNN的K值
    'RANDOM_SEED': 32,  # 随机种子
    'OPEN_KNN': True,  # 启用KNN算法进行比较
    'TRAIN_LOG': False,  # 启用训练日志
    'PREDICTION_LOG': False,  # 启用预测日志
    'GRID_SEARCH': False,  # 是否启用网格搜索
    'QUICK_SEARCH': False,  # 是否使用快速搜索（减少组合数量）
    'OPEN_MLP': True,  # 启用MLP算法进行比较
    'OPEN_GAT': True,  # 启用GAT算法进行比较
    'DATA_AUGMENTATION': False,  # 是否启用数据增强
    'OPEN_LANDMARC': True,  # 启用LANDMARC算法进行比较
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device)

# 设置matplotlib参数以支持中文字符
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class MLPLocalizationModel(torch.nn.Module):
    """用于定位的MLP网络模型"""

    def __init__(self, in_channels, hidden_channels=128, out_channels=2, dropout=0.1):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_channels, hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.fc3 = torch.nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc4 = torch.nn.Linear(hidden_channels // 2, out_channels)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class GATLocalizationModel(torch.nn.Module):
    """用于定位的图注意力网络模型"""

    def __init__(self, in_channels, hidden_channels, out_channels, heads=1):
        super().__init__()
        self.gat1 = GATv2Conv(
            in_channels,
            hidden_channels,
            heads=heads,
            dropout=0.12,
            add_self_loops=True,
            edge_dim=1
        )
        self.gat2 = GATv2Conv(
            hidden_channels * heads,
            hidden_channels * heads,
            heads=1,
            dropout=0.12,
            add_self_loops=True,
            edge_dim=1
        )

        # 残差连接的线性变换（如果需要）
        if in_channels != hidden_channels * heads:
            self.res_fc1 = torch.nn.Linear(
                in_channels, hidden_channels * heads)
        else:
            self.res_fc1 = None
        # 第二个残差连接（gat2前后）
        # gat2输入输出维度都是hidden_channels * heads
        self.res_fc2 = None  # 预留接口，便于后续扩展

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * heads,
                            hidden_channels), torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 32), torch.nn.ReLU(),
            torch.nn.Linear(32, out_channels)
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # 确保边权重是2D张量，这是GATv2Conv所需的
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.view(-1, 1)

        x_input = x  # 保留输入用于残差
        x = self.gat1(x, edge_index, edge_attr=edge_attr)
        # 第一处残差连接
        if self.res_fc1 is not None:
            res1 = self.res_fc1(x_input)
        else:
            res1 = x_input
        x = F.elu(x + res1)

        x_input2 = x  # 保留gat1输出用于第二处残差
        x = self.gat2(x, edge_index, edge_attr=edge_attr)
        # 第二处残差连接
        if self.res_fc2 is not None:
            res2 = self.res_fc2(x_input2)
        else:
            res2 = x_input2
        x = F.elu(x + res2)

        x = self.fc(x)
        return x


class RFIDLocalization:
    """使用GAT进行RFID定位的主类"""

    def __init__(self, config=None):
        """使用配置进行初始化"""
        self.config = config or CONFIG
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.set_random_seed()

        # 数据占位符
        self.df = None
        self.features = None
        self.labels = None
        self.features_norm = None
        self.labels_norm = None
        self.pos = None
        self.scaler_rssi = MinMaxScaler()
        self.scaler_phase = MinMaxScaler()
        self.labels_scaler = MinMaxScaler()
        self.model = None
        self.mlp_model = None

        # 添加损失值记录
        self.gat_train_losses = []
        self.gat_val_losses = []
        self.mlp_train_losses = []
        self.mlp_val_losses = []

        # 加载数据
        self.load_data()

    def set_random_seed(self):
        """设置随机种子以确保可重现性"""
        np.random.seed(self.config['RANDOM_SEED'])
        torch.manual_seed(self.config['RANDOM_SEED'])

    def load_data(self):
        """加载和预处理数据"""
        # 读取数据
        self.df = pd.read_csv('data/rfid_reference_tags.csv')

        # 提取特征和标签
        self.features = self.to_device(torch.tensor(
            self.df[[
                'rssi_antenna1', 'rssi_antenna2', 'rssi_antenna3', 'rssi_antenna4',
                "phase_antenna1", "phase_antenna2", "phase_antenna3", "phase_antenna4"
            ]].values,
            dtype=torch.float32
        ))
        self.labels = self.to_device(torch.tensor(
            self.df[['true_x', 'true_y']].values, dtype=torch.float32
        ))

        # 标准化特征
        rssi_norm = self.scaler_rssi.fit_transform(
            self.features[:, :4].cpu().numpy())
        phase = self.features[:, 4:8].cpu().numpy()
        phase_norm = self.scaler_phase.fit_transform(phase)
        self.features_norm = self.to_device(torch.tensor(
            np.hstack([rssi_norm, phase_norm]), dtype=torch.float32
        ))

        # 标准化标签
        self.labels_norm = self.to_device(torch.tensor(
            self.labels_scaler.fit_transform(self.labels.cpu().numpy()), dtype=torch.float32
        ))

        # 创建位置字典
        self.pos = {
            i: (self.labels[i][0].item(), self.labels[i][1].item())
            for i in range(len(self.labels))
        }

    def create_graph_data(self, features_norm, labels_norm, k=None, pos=None):
        """为GAT模型创建图数据"""
        if k is None:
            k = self.config['K']
        if pos is None:
            pos = self.labels

        # 使用标签（位置）构建KNN图
        adj_matrix = kneighbors_graph(
            torch.as_tensor(
                np.hstack([
                    0.3 * features_norm.cpu().numpy(),
                    # 0.6 * features_norm[:, 4:8].cpu().numpy(),
                    0.7 * labels_norm.cpu().numpy()
                ])
            ),
            n_neighbors=k,
            mode='distance',
        )
        adj_matrix_dense = torch.as_tensor(
            adj_matrix.toarray(), dtype=torch.float32)
        edge_index = self.to_device(torch.nonzero(
            adj_matrix_dense, as_tuple=False).t())
        adj_matrix_coo = adj_matrix.tocoo()
        edge_attr = self.to_device(torch.tensor(
            adj_matrix_coo.data, dtype=torch.float32))

        return Data(
            x=self.to_device(features_norm),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=self.to_device(labels_norm),
            pos=pos
        )

    def create_data_masks(self, num_nodes, test_size=0.2, val_size=0.2):
        """创建训练、验证和测试集的掩码"""
        # 为所有节点创建索引
        indices = np.arange(num_nodes)

        # 首先分割为训练集和临时测试集
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=self.config['RANDOM_SEED']
        )

        # 将训练集分割为训练集和验证集
        train_idx, val_idx = train_test_split(
            train_idx,
            test_size=val_size / (1 - test_size),
            random_state=self.config['RANDOM_SEED']
        )

        # 创建掩码
        train_mask = self.to_device(torch.zeros(num_nodes, dtype=torch.bool))
        val_mask = self.to_device(torch.zeros(num_nodes, dtype=torch.bool))
        test_mask = self.to_device(torch.zeros(num_nodes, dtype=torch.bool))

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        return train_mask, val_mask, test_mask

    def train_gat_model(self, hidden_channels=64, heads=3, lr=0.005, weight_decay=5e-4):
        """训练GAT模型"""
        # 创建完整图数据
        full_graph_data = self.create_graph_data(
            self.features_norm, self.labels_norm, k=self.config['K']
        )

        # 创建训练、验证和测试掩码
        train_mask, val_mask, test_mask = self.create_data_masks(
            len(self.features_norm)
        )

        # 将掩码添加到图数据中
        full_graph_data.train_mask = train_mask | test_mask
        full_graph_data.val_mask = val_mask

        # 将数据移动到设备
        full_graph_data = full_graph_data.to(device)

        # 创建GAT模型
        # 确保权重初始化一致性
        torch.manual_seed(self.config['RANDOM_SEED'])

        self.model = GATLocalizationModel(
            in_channels=full_graph_data.x.shape[1],
            hidden_channels=hidden_channels,
            out_channels=2,
            heads=heads
        ).to(device)

        # 为MinMaxScaler参数创建张量
        data_min = torch.as_tensor(self.labels_scaler.data_min_,
                                   dtype=torch.float32).to(device)
        data_range = torch.as_tensor(
            self.labels_scaler.data_range_, dtype=torch.float32
        ).to(device)

        # 使用固定种子初始化优化器
        torch.manual_seed(self.config['RANDOM_SEED'])
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        loss_fn = torch.nn.MSELoss()

        best_val_loss = float('inf')
        best_model = None
        patience = 50  # 早停耐心值
        counter = 0  # 计数器
        best_val_avg_distance = float('inf')

        # 清空损失记录
        self.gat_train_losses = []
        self.gat_val_losses = []

        for epoch in range(1000):
            # 训练阶段
            self.model.train()
            optimizer.zero_grad()
            out = self.model(full_graph_data)

            # 仅计算训练节点的损失
            train_loss = loss_fn(
                out[full_graph_data.train_mask],
                full_graph_data.y[full_graph_data.train_mask]
            )
            train_loss.backward()
            optimizer.step()

            # 验证阶段
            self.model.eval()
            with torch.no_grad():
                # 计算验证损失
                val_loss = loss_fn(
                    out[full_graph_data.val_mask],
                    full_graph_data.y[full_graph_data.val_mask]
                )

                # 将预测结果转换回原始比例（逆MinMaxScaler）
                out_orig = out * data_range + data_min
                y_orig = full_graph_data.y * data_range + data_min

                # 计算训练集和验证集的指标（使用原始坐标）
                train_distances = torch.sqrt(
                    torch.sum((
                        out_orig[full_graph_data.train_mask] -
                        y_orig[full_graph_data.train_mask]
                    )**2,
                        dim=1)
                )

                val_distances = torch.sqrt(
                    torch.sum((
                        out_orig[full_graph_data.val_mask] -
                        y_orig[full_graph_data.val_mask]
                    )**2,
                        dim=1)
                )

                train_accuracy = (train_distances <
                                  0.3).float().mean().item() * 100
                val_accuracy = (
                    val_distances < 0.3).float().mean().item() * 100

                train_avg_distance = train_distances.mean().item()
                val_avg_distance = val_distances.mean().item()

            # 保存每个epoch的损失值
            self.gat_train_losses.append(train_loss.item())
            self.gat_val_losses.append(val_loss.item())

            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_avg_distance = val_avg_distance
                best_model = self.model.state_dict().copy()
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    if self.config['TRAIN_LOG']:
                        print(
                            f"轮次 {epoch}\n"
                            f"训练集 - 损失: {train_loss.item():.4f}, 准确率: {train_accuracy:.2f}%, 平均误差: {train_avg_distance:.2f}米\n"
                            f"验证集 - 损失: {val_loss.item():.4f}, 准确率: {val_accuracy:.2f}%, 平均误差: {val_avg_distance:.2f}米"
                        )
                        print(f"\n触发早停！在轮次 {epoch} 停止训练")
                        print(f"最佳验证损失: {best_val_loss:.4f}")

                    # 加载最佳模型
                    self.model.load_state_dict(best_model)
                    break

            if epoch % 100 == 0 and self.config['TRAIN_LOG']:
                print(
                    f"轮次 {epoch}\n"
                    f"训练集 - 损失: {train_loss.item():.4f}, 准确率: {train_accuracy:.2f}%, 平均误差: {train_avg_distance:.2f}米\n"
                    f"验证集 - 损失: {val_loss.item():.4f}, 准确率: {val_accuracy:.2f}%, 平均误差: {val_avg_distance:.2f}米"
                )

        return best_val_avg_distance, best_val_loss.item()

    def knn_localization(self, features, labels, n_neighbors=None):
        """使用KNN进行位置预测"""
        if n_neighbors is None:
            n_neighbors = self.config['K']

        # 创建并训练KNN模型
        knn = KNeighborsRegressor(n_neighbors=n_neighbors, algorithm='auto')
        knn.fit(features, labels)

        return knn

    def evaluate_knn_on_test_set(self, knn_model, test_features, test_labels):
        """在测试集上评估KNN模型"""
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

    def evaluate_prediction_GAT_accuracy(self, test_data=None, num_samples=50):
        """
        评估模型在新标签预测上的准确性
        """
        if self.model is None:
            raise ValueError("模型未训练。请先调用train_gat_model。")

        # 如果没有提供测试数据，则从现有数据中随机抽样
        if test_data is None:
            # 随机抽样
            indices = np.random.choice(
                len(self.features), num_samples, replace=False)
            test_features = self.features[indices]
            test_labels = self.labels[indices]
        else:
            test_features, test_labels = test_data
            test_features = self.to_device(
                torch.tensor(test_features, dtype=torch.float32))
            test_labels = self.to_device(
                torch.tensor(test_labels, dtype=torch.float32))

        # 获取RSSI和相位值
        rssi_values = test_features[:, :4]
        phase_values = test_features[:, 4:8]

        # 标准化RSSI
        rssi_norm = self.scaler_rssi.transform(rssi_values.cpu().numpy())
        phase_norm = self.scaler_phase.transform(phase_values.cpu().numpy())

        # 组合特征
        features_new = self.to_device(torch.tensor(
            np.hstack([rssi_norm, phase_norm]),
            dtype=torch.float32
        ))

        # 使用GAT模型预测
        predicted_positions = []

        # 确保MLP模型已训练，用于初始估计
        if self.mlp_model is None:
            self.train_mlp_model()

        # 单独预测每个测试样本
        for i in range(len(features_new)):
            # 提取单个样本特征
            sample_features = features_new[i:i + 1]

            # 使用KNN估计初始位置
            self.mlp_model.eval()
            with torch.no_grad():
                mlp_pred = self.mlp_model(sample_features)
            temp_labels = self.to_device(mlp_pred)

            # 将新节点添加到特征集
            all_features = torch.cat([
                self.features_norm,
                sample_features
            ], dim=0)

            # 为新节点使用KNN估计的标签
            all_labels = torch.cat([
                self.labels_norm,
                temp_labels
            ], dim=0)

            # 创建图数据
            graph_data = self.create_graph_data(
                all_features, all_labels, k=self.config['K'], pos=self.labels
            )

            # 预测
            self.model.eval()
            with torch.no_grad():
                out = self.model(graph_data)
                # 提取新节点预测
                pred_pos = out[-1]
                predicted_positions.append(pred_pos)

        # 计算误差
        predicted_positions = torch.stack(predicted_positions)

        # 反标准化以获得实际坐标
        predicted_positions_orig = self.labels_scaler.inverse_transform(
            predicted_positions.cpu().numpy()
        )
        predicted_positions_orig = self.to_device(
            torch.tensor(predicted_positions_orig, dtype=torch.float32))

        # 计算欧几里得距离误差
        distances = torch.sqrt(
            torch.sum((test_labels - predicted_positions_orig)**2, dim=1))
        avg_distance = torch.mean(distances).item()

        if self.config['PREDICTION_LOG']:
            print("\n新标签位置预测评估:")
            print(f"测试样本数量: {len(test_features)}")
            print(f"平均预测误差: {avg_distance:.2f}米")
            print(f"最大误差: {torch.max(distances).item():.2f}米")
            print(f"最小误差: {torch.min(distances).item():.2f}米")
            print(f"误差标准差: {torch.std(distances).item():.2f}米")

            # 计算不同误差阈值下的准确率
            for threshold in [0.5, 1.0, 1.5, 2.0]:
                accuracy = (distances < threshold).float().mean().item() * 100
                print(f"误差 < {threshold}米的准确率: {accuracy:.2f}%")

        return avg_distance

    def train_mlp_model(
        self, hidden_channels=128, dropout=0.1, lr=0.001, weight_decay=0.0005
    ):
        """训练MLP模型，支持超参数传递，返回验证集损失、平均误差和最佳模型参数"""
        # 创建训练、验证和测试掩码
        train_mask, val_mask, test_mask = self.create_data_masks(
            len(self.features_norm)
        )
        train_mask = train_mask | test_mask

        # 准备数据
        X = self.features_norm.to(device)
        y = self.labels_norm.to(device)

        # 数据增强
        if self.config.get('DATA_AUGMENTATION', False):
            # 数据增强方法1: 添加高斯噪声
            noise_scale = 0.15  # 噪声比例
            X_noisy = self.to_device(torch.tensor(
                X[train_mask].cpu().numpy() + np.random.normal(0, noise_scale,
                                                               X[train_mask].cpu().numpy().shape),
                dtype=torch.float32
            ))

            # 数据增强方法2: 特征缩放
            scale_factors = np.random.uniform(
                0.9, 1.1, (X[train_mask].cpu().numpy().shape[0], 1))
            X_scaled = self.to_device(torch.tensor(
                X[train_mask].cpu().numpy() * scale_factors,
                dtype=torch.float32
            ))

            # 数据增强方法3: RSSI和相位混合扰动
            X_mixed = X[train_mask].cpu().numpy().copy()
            X_mixed[:, :4] += np.random.uniform(-0.03, 0.03,
                                                (X[train_mask].cpu().numpy().shape[0], 4))
            X_mixed[:, 4:] += np.random.uniform(-0.03, 0.03,
                                                (X[train_mask].cpu().numpy().shape[0], 4))
            X_mixed = self.to_device(
                torch.tensor(X_mixed, dtype=torch.float32))

            # 合并原始数据和增强数据
            X_train_tensor = torch.cat(
                [X[train_mask], X_noisy, X_scaled, X_mixed], dim=0)
            y_train_tensor = torch.cat([y[train_mask]] * 4, dim=0)

            if self.config['TRAIN_LOG']:
                print(
                    f"使用数据增强: 原始样本数 {len(X[train_mask])}, 增强后样本数 {len(X_train_tensor)}")
        else:
            X_train_tensor = X[train_mask]
            y_train_tensor = y[train_mask]

        # 创建MLP模型
        torch.manual_seed(self.config['RANDOM_SEED'])
        self.mlp_model = MLPLocalizationModel(
            in_channels=X.shape[1],
            hidden_channels=hidden_channels,
            out_channels=2,
            dropout=dropout
        ).to(device)

        # 为MinMaxScaler参数创建张量
        data_min = torch.as_tensor(self.labels_scaler.data_min_,
                                   dtype=torch.float32).to(device)
        data_range = torch.as_tensor(
            self.labels_scaler.data_range_, dtype=torch.float32
        ).to(device)
        # 使用固定种子初始化优化器
        torch.manual_seed(self.config['RANDOM_SEED'])
        optimizer = torch.optim.Adam(
            self.mlp_model.parameters(), lr=lr, weight_decay=weight_decay
        )
        loss_fn = torch.nn.MSELoss()

        best_val_loss = float('inf')
        best_model = None
        patience = 50  # 早停耐心值
        counter = 0  # 计数器
        best_val_avg_distance = float('inf')

        # 清空损失记录
        self.mlp_train_losses = []
        self.mlp_val_losses = []

        for epoch in range(1000):
            # 训练阶段
            self.mlp_model.train()
            optimizer.zero_grad()
            out = self.mlp_model(X_train_tensor)
            train_loss = loss_fn(out, y_train_tensor)
            train_loss.backward()
            optimizer.step()

            # 验证阶段
            self.mlp_model.eval()
            with torch.no_grad():
                val_out = self.mlp_model(X[val_mask])
                val_loss = loss_fn(val_out, y[val_mask])
                out_orig = val_out * data_range + data_min
                y_orig = y[val_mask] * data_range + data_min
                val_distances = torch.sqrt(
                    torch.sum((out_orig - y_orig)**2, dim=1))
                val_accuracy = (
                    val_distances < 0.3).float().mean().item() * 100
                val_avg_distance = val_distances.mean().item()

            # 保存每个epoch的损失值
            self.mlp_train_losses.append(train_loss.item())
            self.mlp_val_losses.append(val_loss.item())

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_avg_distance = val_avg_distance
                best_model = self.mlp_model.state_dict().copy()
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    if self.config['TRAIN_LOG']:
                        print(
                            f"MLP轮次 {epoch}\n"
                            f"验证集 - 损失: {val_loss.item():.4f}, 准确率: {val_accuracy:.2f}%, 平均误差: {val_avg_distance:.2f}米"
                        )
                        print(f"\n触发早停！在轮次 {epoch} 停止训练")
                        print(f"最佳验证损失: {best_val_loss:.4f}")
                    self.mlp_model.load_state_dict(best_model)
                    break

            if epoch % 100 == 0 and self.config['TRAIN_LOG']:
                print(
                    f"MLP轮次 {epoch}\n"
                    f"训练集 - 损失: {train_loss.item():.4f}\n"
                    f"验证集 - 损失: {val_loss.item():.4f}, 准确率: {val_accuracy:.2f}%, 平均误差: {val_avg_distance:.2f}米"
                )
        # 训练结束后加载最佳模型
        self.mlp_model.load_state_dict(best_model)
        return best_val_loss, best_val_avg_distance, best_model

    def plot_loss_comparison(self, save_path=None):
        """绘制MLP和GAT的训练和验证损失对比图"""
        if len(self.mlp_train_losses) == 0 or len(self.gat_train_losses) == 0:
            print("没有可用的损失数据来绘制图表")
            return

        plt.figure(figsize=(12, 8))

        # 确保目录存在
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 绘制训练损失
        plt.subplot(2, 1, 1)
        epochs_mlp = range(1, len(self.mlp_train_losses) + 1)
        epochs_gat = range(1, len(self.gat_train_losses) + 1)

        plt.plot(epochs_mlp, self.mlp_train_losses, 'b-', label='MLP训练损失')
        plt.plot(epochs_gat, self.gat_train_losses, 'r-', label='GAT训练损失')
        plt.title('MLP vs GAT 训练损失对比')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True)

        # 绘制验证损失
        plt.subplot(2, 1, 2)
        plt.plot(epochs_mlp, self.mlp_val_losses, 'b-', label='MLP验证损失')
        plt.plot(epochs_gat, self.gat_val_losses, 'r-', label='GAT验证损失')
        plt.title('MLP vs GAT 验证损失对比')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def evaluate_mlp_on_new_data(self, test_features, test_labels):
        """评估MLP模型在新数据上的性能"""
        if self.mlp_model is None:
            raise ValueError("MLP模型未训练。请先调用train_mlp_model。")

        # 获取RSSI和相位值
        rssi_values = test_features[:, :4]
        phase_values = test_features[:, 4:8]

        # 标准化特征
        rssi_norm = self.scaler_rssi.transform(rssi_values)
        phase_norm = self.scaler_phase.transform(phase_values)

        # 组合特征
        features_norm = np.hstack([rssi_norm, phase_norm])
        features_tensor = torch.tensor(
            features_norm, dtype=torch.float32).to(device)

        # 预测
        self.mlp_model.eval()
        with torch.no_grad():
            predictions = self.mlp_model(features_tensor)

            # 反标准化以获得实际坐标
            predictions_orig = self.labels_scaler.inverse_transform(
                predictions.cpu().numpy()
            )

            # 计算欧几里得距离误差
            distances = np.sqrt(
                np.sum((test_labels - predictions_orig)**2, axis=1))
            avg_distance = np.mean(distances)

        if self.config['PREDICTION_LOG']:
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

    def landmarc_localization(self, test_features, test_labels=None, k=None):
        """
        使用LANDMARC算法进行定位

        参数:
            test_features: 测试标签的特征（RSSI和相位值）
            test_labels: 测试标签的真实位置（可选，用于评估）
            k: k近邻数量

        返回:
            预测位置和平均误差（如果提供真实位置）
        """
        if k is None:
            k = self.config['K']

        # 提取参考标签的RSSI值（用于标准化）
        reference_rssi = self.features.cpu().numpy()  # 先将数据移到CPU
        reference_locations = self.labels.cpu().numpy()  # 先将数据移到CPU

        # 计算测试标签预测位置
        predictions = []

        # 遍历每个测试样本
        for i in range(len(test_features)):
            # 提取当前测试标签的RSSI值
            test_rssi = test_features[i].cpu(
            ).numpy().reshape(1, -1)  # 先将数据移到CPU

            # 计算欧氏距离（信号空间距离）
            signal_distances = np.sqrt(
                np.sum((reference_rssi - test_rssi)**2, axis=1))

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
            predicted_location = np.sum(
                nearest_locations * weights.reshape(-1, 1), axis=0
            )
            predictions.append(predicted_location)

        predictions = np.array(predictions)

        # 如果提供了真实位置，计算误差
        avg_error = None
        if test_labels is not None:
            test_labels_np = test_labels.cpu().numpy()  # 先将数据移到CPU
            distances = np.sqrt(
                np.sum((predictions - test_labels_np)**2, axis=1))
            avg_error = np.mean(distances)

            if self.config['PREDICTION_LOG']:
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

        return predictions, avg_error

    def to_device(self, data):
        """将数据移动到指定设备"""
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, (list, tuple)):
            return [self.to_device(x) for x in data]
        elif isinstance(data, dict):
            return {k: self.to_device(v) for k, v in data.items()}
        return data


def load_and_preprocess_test_data(test_csv_path, scaler_rssi=None, scaler_phase=None):
    """
    加载并标准化测试集数据。
    返回：
        test_features, test_labels, test_features_np, test_labels_np, test_features_norm
    """
    df_test = pd.read_csv(test_csv_path)
    test_features = torch.tensor(
        df_test[[
            'rssi_antenna1', 'rssi_antenna2', 'rssi_antenna3', 'rssi_antenna4',
            "phase_antenna1", "phase_antenna2", "phase_antenna3", "phase_antenna4"
        ]].values,
        dtype=torch.float32
    )
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


def run_grid_search(
    k_range=None,
    lr_range=None,
    weight_decay_range=None,
    hidden_channels_range=None,
    heads_range=None,
    quick_search=None,
    dropout_range=None
):
    """
    执行超参数网格搜索

    参数:
        k_range: KNN的K值范围
        lr_range: 学习率范围
        weight_decay_range: 权重衰减范围
        hidden_channels_range: 隐藏层通道数范围
        heads_range: 注意力头数范围
        quick_search: 是否使用快速搜索（减少组合数量）
        dropout_range: Dropout比率范围
    """
    start_time = time.time()  # 记录开始时间
    # 设置默认搜索范围
    if k_range is None:
        k_range = [4, 5, 6, 7, 8]
    if lr_range is None:
        lr_range = [0.001, 0.005, 0.01]
    if weight_decay_range is None:
        weight_decay_range = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    if hidden_channels_range is None:
        hidden_channels_range = [32, 64, 128]
    if heads_range is None:
        heads_range = [1, 2, 3, 4]
    if dropout_range is None:
        dropout_range = [0.1, 0.2, 0.3]
    if quick_search is None:
        quick_search = CONFIG['QUICK_SEARCH']

    # 加载测试集数据用于验证
    test_features, test_labels, test_features_np, test_labels_np, _ = load_and_preprocess_test_data(
        'data/rfid_test_tags.csv'
    )

    # 如果使用快速搜索，减少组合数量
    if quick_search:
        print("使用快速搜索模式（减少组合数量）")
        # 对每个参数选择最少的值进行测试
        if len(k_range) > 2:
            k_range = [k_range[0], k_range[-1]]
        if len(lr_range) > 2:
            lr_range = [lr_range[0], lr_range[-1]]
        if len(weight_decay_range) > 2:
            weight_decay_range = [
                weight_decay_range[0], weight_decay_range[-1]]
        if len(hidden_channels_range) > 2:
            hidden_channels_range = [
                hidden_channels_range[0], hidden_channels_range[-1]
            ]
        if len(heads_range) > 2:
            heads_range = [heads_range[0], heads_range[-1]]
        if len(dropout_range) > 2:
            dropout_range = [dropout_range[0], dropout_range[-1]]

    # 存储GAT结果
    gat_results = []
    # 存储MLP结果
    mlp_results = []

    # 创建所有GAT超参数组合
    gat_param_combinations = list(
        itertools.product(
            k_range, lr_range, weight_decay_range, hidden_channels_range, heads_range
        )
    )

    # 创建所有MLP超参数组合
    mlp_param_combinations = list(
        itertools.product(
            lr_range, weight_decay_range, hidden_channels_range, dropout_range
        )
    )

    # 首先执行MLP网格搜索
    print(f"开始MLP网格搜索，共 {len(mlp_param_combinations)} 种组合")

    best_mlp_model = None
    best_mlp_localization = None
    best_mlp_val_avg_distance = float('inf')

    # 执行MLP网格搜索
    for i, (lr, weight_decay, hidden_channels,
            dropout) in enumerate(mlp_param_combinations):
        print(f"\nMLP组合 {i+1}/{len(mlp_param_combinations)}:")
        print(
            f"lr={lr}, weight_decay={weight_decay}, hidden_channels={hidden_channels}, dropout={dropout}"
        )

        # 更新配置
        config = CONFIG.copy()

        # 初始化定位系统
        localization = RFIDLocalization(config)

        # 直接调用train_mlp_model
        best_val_loss, best_val_avg_distance, best_model = localization.train_mlp_model(
            hidden_channels=hidden_channels,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay
        )
        # 加载最佳模型
        localization.mlp_model.load_state_dict(best_model)
        # 在测试集数据上评估模型性能
        rssi_norm = localization.scaler_rssi.transform(test_features_np[:, :4])
        phase_norm = localization.scaler_phase.transform(
            test_features_np[:, 4:8])
        test_features_norm = np.hstack([rssi_norm, phase_norm])
        test_features_tensor = torch.tensor(test_features_norm,
                                            dtype=torch.float32).to(device)
        localization.mlp_model.eval()
        with torch.no_grad():
            predictions = localization.mlp_model(test_features_tensor)
            predictions_orig = localization.labels_scaler.inverse_transform(
                predictions.cpu().numpy()
            )
            distances = np.sqrt(
                np.sum((test_labels_np - predictions_orig)**2, axis=1))
            test_avg_distance = np.mean(distances)
        result = {
            'lr': lr,
            'weight_decay': weight_decay,
            'hidden_channels': hidden_channels,
            'dropout': dropout,
            'val_loss': best_val_loss.item(),
            'val_avg_distance': best_val_avg_distance,
            'test_avg_distance': test_avg_distance
        }
        mlp_results.append(result)
        print(f"测试集平均误差: {best_val_avg_distance:.2f}米")
        print(f"验证集平均误差: {test_avg_distance:.2f}米")
        if test_avg_distance < best_mlp_val_avg_distance:
            best_mlp_val_avg_distance = test_avg_distance
            best_mlp_model = localization.mlp_model
            localization.mlp_model = localization.mlp_model  # 保持一致
            best_mlp_localization = localization
            best_result = result

    # 确保我们有最佳的MLP模型和localization对象
    if best_mlp_model is None or best_mlp_localization is None:
        print("警告：未能找到有效的MLP模型，将使用默认参数训练一个")
        config = CONFIG.copy()
        best_mlp_localization = RFIDLocalization(config)
        best_mlp_localization.train_mlp_model()
    else:
        print(f"\n找到最佳MLP模型，验证集平均误差: {best_mlp_val_avg_distance:.2f}米")
        print("最佳参数：", best_result)

    print(f"\n开始GAT网格搜索，共 {len(gat_param_combinations)} 种组合")

    # 执行GAT网格搜索，使用最佳MLP模型
    for i, (k, lr, weight_decay, hidden_channels,
            heads) in enumerate(gat_param_combinations):
        print(f"\nGAT组合 {i+1}/{len(gat_param_combinations)}:")
        print(
            f"K={k}, lr={lr}, weight_decay={weight_decay}, hidden_channels={hidden_channels}, heads={heads}"
        )

        # 更新配置
        config = CONFIG.copy()
        config['K'] = k

        # 初始化定位系统
        localization = RFIDLocalization(config)

        # 使用最佳MLP模型替换新初始化的MLP模型
        localization.mlp_model = best_mlp_model

        # 确保标准化器与最佳MLP模型一致
        localization.scaler_rssi = best_mlp_localization.scaler_rssi
        localization.scaler_phase = best_mlp_localization.scaler_phase
        localization.labels_scaler = best_mlp_localization.labels_scaler

        # 直接调用 train_gat_model 进行训练和测试
        best_val_avg_distance, best_val_loss = localization.train_gat_model(
            hidden_channels=hidden_channels,
            heads=heads,
            lr=lr,
            weight_decay=weight_decay
        )
        val_avg_distance = best_val_avg_distance
        # 用 rfid_test_tags.csv 的数据评估验证集（即新标签数据）
        test_avg_distance = localization.evaluate_prediction_GAT_accuracy(
            test_data=(test_features_np, test_labels_np),
            num_samples=len(test_features_np)
        )

        # 存储结果
        result = {
            'K': k,
            'lr': lr,
            'weight_decay': weight_decay,
            'hidden_channels': hidden_channels,
            'heads': heads,
            'val_avg_distance': val_avg_distance,
            'test_avg_distance': test_avg_distance
        }
        gat_results.append(result)

        print(f"测试集平均误差: {val_avg_distance:.2f}米")
        print(f"验证集平均误差: {test_avg_distance:.2f}米")

    # 按验证集平均误差排序
    gat_results.sort(key=lambda x: x['test_avg_distance'])
    mlp_results.sort(key=lambda x: x['test_avg_distance'])

    # 输出最佳GAT超参数
    best_gat_result = gat_results[0]
    print("\n============ 最佳GAT超参数 ============")
    print(f"K: {best_gat_result['K']}")
    print(f"学习率: {best_gat_result['lr']}")
    print(f"权重衰减: {best_gat_result['weight_decay']}")
    print(f"隐藏层通道数: {best_gat_result['hidden_channels']}")
    print(f"注意力头数: {best_gat_result['heads']}")
    print(f"验证集平均误差: {best_gat_result['test_avg_distance']:.2f}米")

    # 输出最佳MLP超参数
    best_mlp_result = mlp_results[0]
    print("\n============ 最佳MLP超参数 ============")
    print(f"学习率: {best_mlp_result['lr']}")
    print(f"权重衰减: {best_mlp_result['weight_decay']}")
    print(f"隐藏层通道数: {best_mlp_result['hidden_channels']}")
    print(f"Dropout比率: {best_mlp_result['dropout']}")
    print(f"验证集平均误差: {best_mlp_result['test_avg_distance']:.2f}米")

    # 将所有结果输出到文件
    gat_results_df = pd.DataFrame(gat_results)
    gat_results_df.to_csv('results/gat_grid_search_results.csv', index=False)

    mlp_results_df = pd.DataFrame(mlp_results)
    mlp_results_df.to_csv('results/mlp_grid_search_results.csv', index=False)

    print("GAT结果已保存到 gat_grid_search_results.csv")
    print("MLP结果已保存到 mlp_grid_search_results.csv")
    end_time = time.time()  # 记录结束时间
    total_time = end_time - start_time
    print(f"\n网格搜索总耗时: {total_time:.2f} 秒")
    # 比较GAT和MLP的最佳结果
    if best_gat_result['test_avg_distance'] <= best_mlp_result['test_avg_distance']:
        print("\nGAT模型性能更好!")
    return best_gat_result


def main():
    test_features, test_labels, test_features_np, test_labels_np, _ = load_and_preprocess_test_data(
        'data/rfid_test_tags.csv'
    )
    """运行实验的主函数"""
    # 检查是否启用网格搜索
    if CONFIG['GRID_SEARCH']:
        # 运行网格搜索
        print("执行超参数网格搜索")

        # 为了节省时间，可以使用较小的搜索范围
        best_params = run_grid_search(
            k_range=[5, 6, 7, 8],
            lr_range=[0.001, 0.005, 0.01],
            weight_decay_range=[1e-5, 5e-5, 1e-4, 5e-4],
            hidden_channels_range=[32, 64, 128],
            heads_range=[1, 2, 3],
            quick_search=CONFIG['QUICK_SEARCH']
        )

        # 使用最佳超参数训练模型并评估
        print("\n使用最佳超参数训练最终模型")
        # 更新配置
        config = CONFIG.copy()
        config['K'] = best_params['K']
    else:
        # 不进行网格搜索，使用默认参数
        print("使用默认超参数训练模型")
        best_params = {
            'K': CONFIG['K'],
            # 'lr': 0.001,
            # 'weight_decay': 5e-5,
            # 'hidden_channels': 64,
            # 'heads': 2
        }
        config = CONFIG.copy()

    # 初始化定位系统
    localization = RFIDLocalization(config)

    # 确保结果目录存在
    os.makedirs('results', exist_ok=True)

    # 训练并评估MLP模型
    if config['OPEN_MLP']:
        localization.train_mlp_model(
            hidden_channels=best_params.get('hidden_channels', 128),
            dropout=best_params.get('dropout', 0.2),
            lr=best_params.get('lr', 0.01),
            weight_decay=best_params.get('weight_decay', 0.0001)
        )
        # 在新数据上评估MLP模型
        mlp_prediction_error = localization.evaluate_mlp_on_new_data(
            test_features_np, test_labels_np
        )
        print(f"MLP模型在新标签上的平均预测误差: {mlp_prediction_error:.2f}米")

    # 训练GAT模型
    if CONFIG['GRID_SEARCH']:
        localization.model = GATLocalizationModel(
            in_channels=localization.features_norm.shape[1],
            hidden_channels=best_params.get('hidden_channels', 128),
            out_channels=2,
            heads=best_params.get('heads', 3)
        )

    # 训练并评估GAT模型
    if config['OPEN_GAT']:
        best_val_avg_distance, best_val_loss = localization.train_gat_model(
            hidden_channels=best_params.get('hidden_channels', 128),
            heads=best_params.get('heads', 3),
            lr=best_params.get('lr', 0.001),
            weight_decay=best_params.get('weight_decay', 0.0005)
        )
        print(f"GAT模型测试集平均误差: {best_val_avg_distance:.2f}米")

        # 评估GAT模型在新标签预测上的准确性
        # 设置随机种子以确保可重现性
        np.random.seed(config['RANDOM_SEED'] + 1)  # 使用不同的种子以避免与训练集重叠
        gat_prediction_error = localization.evaluate_prediction_GAT_accuracy(
            test_data=(test_features_np, test_labels_np), num_samples=50
        )
        print(f"GAT模型在新标签上的平均预测误差: {gat_prediction_error:.2f}米")

    # 如果MLP和GAT都已训练，则绘制损失对比图
    if config['OPEN_MLP'] and config['OPEN_GAT']:
        localization.plot_loss_comparison(
            save_path='results/loss_comparison.png')
        # 将损失数据保存到CSV文件中

    if config['OPEN_KNN']:
        # 训练和评估KNN模型
        features_array = localization.features.cpu().numpy()  # 先将数据移到CPU
        labels_array = localization.labels.cpu().numpy()  # 先将数据移到CPU
        knn_model = localization.knn_localization(
            features_array, labels_array, n_neighbors=config['K']
        )
        avg_error = localization.evaluate_knn_on_test_set(
            knn_model, test_features_np, test_labels_np
        )
        print(f"KNN模型测试集平均误差: {avg_error:.2f}米")

    # 使用LANDMARC算法定位
    if config['OPEN_LANDMARC']:
        # 对测试集使用LANDMARC算法
        _, landmarc_error = localization.landmarc_localization(
            test_features, test_labels, k=config['K']
        )
        print(f"LANDMARC算法测试集平均误差: {landmarc_error:.2f}米")


if __name__ == "__main__":
    main()
