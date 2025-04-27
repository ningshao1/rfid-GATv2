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

# 忽略警告
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 配置
CONFIG = {
    'K': 2,  # KNN的K值
    'RANDOM_SEED': 32,  # 随机种子
    'OPEN_KNN': True,  # 启用KNN算法进行比较
    'TRAIN_LOG': False  # 启用训练日志
}

# 设置matplotlib参数以支持中文字符
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class GATLocalizationModel(torch.nn.Module):
    """用于定位的图注意力网络模型"""

    def __init__(self, in_channels, hidden_channels, out_channels, heads=1):
        super().__init__()
        self.gat1 = GATv2Conv(
            in_channels,
            hidden_channels,
            heads=heads,
            dropout=0.1,
            add_self_loops=False,
            edge_dim=1
        )
        self.gat2 = GATv2Conv(
            hidden_channels * heads,
            hidden_channels * heads,
            heads=1,
            dropout=0.1,
            add_self_loops=False,
            edge_dim=1
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * heads, hidden_channels), torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 32), torch.nn.ReLU(),
            torch.nn.Linear(32, out_channels)
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # 确保边权重是2D张量，这是GATv2Conv所需的
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.view(-1, 1)

        x = self.gat1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = self.gat2(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = self.fc(x)
        return x


class RFIDLocalization:
    """使用GAT进行RFID定位的主类"""

    def __init__(self, config=None):
        """使用配置进行初始化"""
        self.config = config or CONFIG
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
        self.features = torch.tensor(
            self.df[[
                'rssi_antenna1', 'rssi_antenna2', 'rssi_antenna3', 'rssi_antenna4',
                "phase_antenna1", "phase_antenna2", "phase_antenna3", "phase_antenna4"
            ]].values,
            dtype=torch.float32
        )
        self.labels = torch.tensor(
            self.df[['true_x', 'true_y']].values, dtype=torch.float32
        )

        # 标准化特征
        rssi_norm = self.scaler_rssi.fit_transform(self.features[:, :4])
        phase = self.features[:, 4:8]
        phase_norm = self.scaler_phase.fit_transform(phase)
        self.features_norm = torch.tensor(
            np.hstack([rssi_norm, phase_norm]), dtype=torch.float32
        )

        # 标准化标签
        self.labels_norm = torch.tensor(
            self.labels_scaler.fit_transform(self.labels), dtype=torch.float32
        )

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
                np.hstack([0.7 * features_norm[:, 4:8], 0.3 * labels_norm])
            ),
            n_neighbors=k,
            mode='distance',
        )
        adj_matrix_dense = torch.as_tensor(adj_matrix.toarray(), dtype=torch.float32)
        edge_index = torch.nonzero(adj_matrix_dense, as_tuple=False).t()
        adj_matrix_coo = adj_matrix.tocoo()
        edge_attr = torch.tensor(adj_matrix_coo.data, dtype=torch.float32)

        return Data(
            features_norm,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=labels_norm,
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
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        return train_mask, val_mask, test_mask

    def train_gat_model(self):
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
        full_graph_data.train_mask = train_mask
        full_graph_data.val_mask = val_mask
        full_graph_data.test_mask = test_mask

        # 检查是否有可用的GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 将数据移动到设备
        full_graph_data = full_graph_data.to(device)

        # 创建GAT模型
        # 确保权重初始化一致性
        torch.manual_seed(self.config['RANDOM_SEED'])

        self.model = GATLocalizationModel(
            in_channels=full_graph_data.x.shape[1],
            hidden_channels=64,
            out_channels=2,
            heads=3
        ).to(device)

        # 为MinMaxScaler参数创建张量
        data_min = torch.as_tensor(self.labels_scaler.data_min_,
                                   dtype=torch.float32).to(device)
        data_range = torch.as_tensor(
            self.labels_scaler.data_range_, dtype=torch.float32
        ).to(device)

        # 使用固定种子初始化优化器
        torch.manual_seed(self.config['RANDOM_SEED'])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005)
        loss_fn = torch.nn.MSELoss()

        best_val_loss = float('inf')
        best_model = None
        patience = 50  # 早停耐心值
        counter = 0  # 计数器

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

                train_accuracy = (train_distances < 0.3).float().mean().item() * 100
                val_accuracy = (val_distances < 0.3).float().mean().item() * 100

                train_avg_distance = train_distances.mean().item()
                val_avg_distance = val_distances.mean().item()

            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
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

        # 测试阶段
        self.model.eval()
        with torch.no_grad():
            # 使用最佳模型进行测试
            test_out = self.model(full_graph_data)
            test_out_orig = test_out[full_graph_data.test_mask] * data_range + data_min
            test_y_orig = full_graph_data.y[full_graph_data.test_mask
                                           ] * data_range + data_min

            # 计算测试集的指标
            test_distances = torch.sqrt(
                torch.sum((test_out_orig - test_y_orig)**2, dim=1)
            )

            test_accuracy = (test_distances < 0.3).float().mean().item() * 100
            test_avg_distance = test_distances.mean().item()

        return test_avg_distance

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
        test_features = np.array(test_features)
        test_labels = np.array(test_labels)

        # 预测测试集
        y_pred = knn_model.predict(test_features)

        # 计算实际米数的误差
        distances = np.sqrt(np.sum((test_labels - y_pred)**2, axis=1))
        avg_distance = np.mean(distances)
        return avg_distance

    def evaluate_prediction_accuracy(
        self, test_data=None, num_samples=50, verbose=True
    ):
        """
        评估模型在新标签预测上的准确性
        """
        if self.model is None:
            raise ValueError("模型未训练。请先调用train_gat_model。")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 如果没有提供测试数据，则从现有数据中随机抽样
        if test_data is None:
            # 随机抽样
            indices = np.random.choice(len(self.features), num_samples, replace=False)
            test_features = self.features[indices].numpy()
            test_labels = self.labels[indices].numpy()
        else:
            test_features, test_labels = test_data

        # 获取RSSI和相位值
        rssi_values = test_features[:, :4]
        phase_values = test_features[:, 4:8]

        # 标准化RSSI
        rssi_norm = self.scaler_rssi.transform(rssi_values)
        phase_norm = self.scaler_phase.transform(phase_values)

        # 组合特征
        features_new = np.hstack([rssi_norm, phase_norm])

        # 创建KNN模型用于初步位置估计
        features_array = np.array(self.features)
        labels_array = np.array(self.labels)
        knn_model = KNeighborsRegressor(
            n_neighbors=self.config['K'],
            algorithm='auto',
        )
        knn_model.fit(features_array, labels_array)

        # 使用GAT模型预测
        predicted_positions = []

        # 单独预测每个测试样本
        for i in range(len(features_new)):
            # 提取单个样本特征
            sample_features = features_new[i:i + 1]

            # 转换为张量
            features_tensor = torch.as_tensor(sample_features,
                                              dtype=torch.float32).to(device)

            # 使用KNN估计初始位置
            knn_pred = knn_model.predict(test_features[i:i + 1])

            temp_labels = torch.as_tensor(
                self.labels_scaler.transform(knn_pred), dtype=torch.float32
            ).to(device)

            # 将新节点添加到特征集
            all_features = torch.cat([
                torch.as_tensor(self.features_norm, dtype=torch.float32),
                features_tensor
            ],
                                     dim=0)

            # 为新节点使用KNN估计的标签
            all_labels = torch.cat([self.labels_norm, temp_labels], dim=0)

            # 创建图数据
            graph_data = self.create_graph_data(
                all_features, all_labels, k=self.config['K'], pos=self.labels
            )

            # 预测
            self.model.eval()
            with torch.no_grad():
                out = self.model(graph_data)
                # 提取新节点预测
                pred_pos = out[-1].cpu().numpy()
                predicted_positions.append(pred_pos)

        # 计算误差
        predicted_positions = np.array(predicted_positions)

        # 反标准化以获得实际坐标
        predicted_positions_orig = self.labels_scaler.inverse_transform(
            predicted_positions
        )

        # 计算欧几里得距离误差
        distances = np.sqrt(np.sum((test_labels - predicted_positions_orig)**2, axis=1))
        avg_distance = np.mean(distances)

        if verbose:
            print("\n新标签位置预测评估:")
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


def main():
    """运行实验的主函数"""
    for k in range(4, 9):  # K从4到8
        print(f"K = {k}")

        # 更新配置
        config = CONFIG.copy()
        config['K'] = k

        # 初始化定位系统
        localization = RFIDLocalization(config)

        # 训练GAT模型
        gat_error = localization.train_gat_model()
        print(f"GAT模型测试集平均误差: {gat_error:.2f}米")

        if config['OPEN_KNN']:
            # 在测试集上评估
            df_test = pd.read_csv('data/rfid_test_tags.csv')
            test_features = torch.tensor(
                df_test[[
                    'rssi_antenna1', 'rssi_antenna2', 'rssi_antenna3', 'rssi_antenna4',
                    "phase_antenna1", "phase_antenna2", "phase_antenna3",
                    "phase_antenna4"
                ]].values,
                dtype=torch.float32
            )
            test_labels = torch.tensor(
                df_test[['true_x', 'true_y']].values, dtype=torch.float32
            )
            test_features_np = np.array(test_features)
            test_labels_np = np.array(test_labels)

            # 训练和评估KNN模型
            features_array = np.array(localization.features)
            labels_array = np.array(localization.labels)
            knn_model = localization.knn_localization(
                features_array[:, 4:8], labels_array, n_neighbors=k
            )
            avg_error = localization.evaluate_knn_on_test_set(
                knn_model, test_features_np[:, 4:8], test_labels_np
            )
            print(f"KNN模型测试集平均误差: {avg_error:.2f}米")

        # 评估GAT模型在新标签预测上的准确性
        # 设置随机种子以确保可重现性
        np.random.seed(config['RANDOM_SEED'] + 1)  # 使用不同的种子以避免与训练集重叠

        gat_prediction_error = localization.evaluate_prediction_accuracy(
            test_data=(test_features_np, test_labels_np), num_samples=50, verbose=False
        )
        print(f"GAT模型在新标签上的平均预测误差: {gat_prediction_error:.2f}米")
        print()


if __name__ == "__main__":
    main()
