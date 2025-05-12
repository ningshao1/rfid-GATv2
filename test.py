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
import logging
from datetime import datetime

# 导入配置
from config import CONFIG

# 导入自定义模型
from models import MLPLocalizationModel, train_mlp_model, evaluate_mlp_on_new_data
from models.utils.data_augmentation import apply_data_augmentation
from models.utils.data_loader import load_and_preprocess_test_data
# 导入KNN和LANDMARC模型
from models.knn.model import knn_localization, evaluate_knn_on_test_set
from models.landmarc.model import landmarc_localization, evaluate_landmarc
# 导入异构图模型
from models.heterogeneous import HeterogeneousGNNModel, create_heterogeneous_graph_data, add_new_node_to_hetero_graph, train_hetero_model as train_hetero_model_func
from models.gat.model import GATLocalizationModel
from models.gat.utils import train_gat_model, evaluate_prediction_GAT_accuracy, create_data_masks, to_device
# 导入utils的run_grid_search
from models.utils.grid_search import run_grid_search

# 忽略警告
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device)

# 设置matplotlib参数以支持中文字符
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class RFIDLocalization:
    """使用GAT进行RFID定位的主类"""

    def __init__(self, config=None):
        """使用配置进行初始化"""
        self.config = config or CONFIG
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.set_random_seed()

        # 设置日志
        os.makedirs('log', exist_ok=True)
        self.log_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 训练日志
        if self.config['TRAIN_LOG']:
            self.train_logger = logging.getLogger('train_log')
            self.train_logger.setLevel(logging.INFO)
            train_handler = logging.FileHandler(
                f'log/train_{self.log_time}.log', encoding='utf-8'
            )
            train_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            self.train_logger.addHandler(train_handler)

        # 预测日志
        if self.config['PREDICTION_LOG']:
            self.prediction_logger = logging.getLogger('prediction_log')
            self.prediction_logger.setLevel(logging.INFO)
            prediction_handler = logging.FileHandler(
                f'log/prediction_{self.log_time}.log', encoding='utf-8'
            )
            prediction_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(message)s')
            )
            self.prediction_logger.addHandler(prediction_handler)

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
        self.hetero_model = None  # 异构图模型

        # 天线位置从配置加载
        self.antenna_locations = torch.tensor(
            self.config['ANTENNA_LOCATIONS'], dtype=torch.float32
        ).to(self.device)

        # 添加损失值记录
        self.gat_train_losses = []
        self.gat_val_losses = []
        self.mlp_train_losses = []
        self.mlp_val_losses = []
        self.hetero_train_losses = []  # 异构图模型训练损失
        self.hetero_val_losses = []  # 异构图模型验证损失

        # 加载数据
        self.load_data()

    def set_random_seed(self):
        """设置随机种子以确保可重现性"""
        np.random.seed(self.config['RANDOM_SEED'])
        torch.manual_seed(self.config['RANDOM_SEED'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config['RANDOM_SEED'])
            torch.cuda.manual_seed_all(self.config['RANDOM_SEED'])
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def load_data(self):
        """加载和预处理数据"""
        # 读取数据
        self.df = pd.read_csv(self.config['REFERENCE_DATA_PATH'])

        # 提取特征和标签
        self.features = to_device(
            torch.tensor(
                self.df[[
                    'rssi_antenna1', 'rssi_antenna2', 'rssi_antenna3', 'rssi_antenna4',
                    "phase_antenna1", "phase_antenna2", "phase_antenna3",
                    "phase_antenna4"
                ]].values,
                dtype=torch.float32
            ), self.device
        )
        self.labels = to_device(
            torch.tensor(self.df[['true_x', 'true_y']].values, dtype=torch.float32),
            self.device
        )

        # 标准化特征
        rssi_norm = self.scaler_rssi.fit_transform(self.features[:, :4].cpu().numpy())
        phase = self.features[:, 4:8].cpu().numpy()
        phase_norm = self.scaler_phase.fit_transform(phase)
        self.features_norm = to_device(
            torch.tensor(np.hstack([rssi_norm, phase_norm]), dtype=torch.float32),
            self.device
        )

        # 标准化标签
        self.labels_norm = to_device(
            torch.tensor(
                self.labels_scaler.fit_transform(self.labels.cpu().numpy()),
                dtype=torch.float32
            ), self.device
        )

        # 创建位置字典
        self.pos = {
            i: (self.labels[i][0].item(), self.labels[i][1].item())
            for i in range(len(self.labels))
        }

    def train_gat_model(self, hidden_channels=64, heads=3, lr=0.005, weight_decay=5e-4):
        return train_gat_model(self, hidden_channels, heads, lr, weight_decay)

    def knn_localization(self, features, labels, n_neighbors=None):
        """使用KNN进行位置预测"""
        if n_neighbors is None:
            n_neighbors = self.config['K']

        # 调用导入的knn_localization函数
        return knn_localization(features, labels, n_neighbors=n_neighbors)

    def evaluate_knn_on_test_set(self, knn_model, test_features, test_labels):
        """在测试集上评估KNN模型"""
        # 调用导入的evaluate_knn_on_test_set函数
        return evaluate_knn_on_test_set(knn_model, test_features, test_labels)

    def evaluate_prediction_GAT_accuracy(self, test_data=None, num_samples=50):
        return evaluate_prediction_GAT_accuracy(self, test_data, num_samples)

    def train_mlp_model(
        self, hidden_channels=128, dropout=0.1, lr=0.001, weight_decay=0.0005
    ):
        """训练MLP模型，支持超参数传递，返回验证集损失、平均误差和最佳模型参数"""
        # 调用models包中的train_mlp_model函数
        return train_mlp_model(
            self,
            hidden_channels=hidden_channels,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay
        )

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
        max_epoch = 50
        epochs_mlp = range(1, min(len(self.mlp_train_losses), max_epoch) + 1)
        epochs_gat = range(1, min(len(self.gat_train_losses), max_epoch) + 1)
        plt.plot(epochs_mlp, self.mlp_train_losses[:max_epoch], 'b-', label='MLP训练损失')
        plt.plot(epochs_gat, self.gat_train_losses[:max_epoch], 'r-', label='GAT训练损失')
        plt.title('MLP vs GAT 训练损失对比')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True)

        # 绘制验证损失
        plt.subplot(2, 1, 2)
        plt.plot(epochs_mlp, self.mlp_val_losses[:max_epoch], 'b-', label='MLP验证损失')
        plt.plot(epochs_gat, self.gat_val_losses[:max_epoch], 'r-', label='GAT验证损失')
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
        # 调用models包中的evaluate_mlp_on_new_data函数
        return evaluate_mlp_on_new_data(self, test_features, test_labels)

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

        # 提取参考标签的特征和位置
        reference_features = self.features
        reference_locations = self.labels

        # 调用导入的landmarc_localization函数
        predictions, avg_error = landmarc_localization(
            reference_features, reference_locations, test_features, test_labels, k
        )

        # 如果启用了预测日志，打印评估信息
        if test_labels is not None and self.config['PREDICTION_LOG']:
            if isinstance(test_labels, torch.Tensor):
                test_labels_np = test_labels.cpu().numpy()
            else:
                test_labels_np = test_labels

            if isinstance(predictions, torch.Tensor):
                predictions_np = predictions.cpu().numpy()
            else:
                predictions_np = predictions

            distances = np.sqrt(np.sum((predictions_np - test_labels_np)**2, axis=1))

            log_message = "\nLANDMARC算法位置预测评估:\n"
            log_message += f"测试样本数量: {len(test_features)}\n"
            log_message += f"平均预测误差: {avg_error:.2f}米\n"
            log_message += f"最大误差: {np.max(distances):.2f}米\n"
            log_message += f"最小误差: {np.min(distances):.2f}米\n"
            log_message += f"误差标准差: {np.std(distances):.2f}米\n"

            # 计算不同误差阈值下的准确率
            for threshold in [0.5, 1.0, 1.5, 2.0]:
                accuracy = np.mean(distances < threshold) * 100
                log_message += f"误差 < {threshold}米的准确率: {accuracy:.2f}%\n"

            # 打印到控制台并写入日志
            self.prediction_logger.info(log_message)

        return predictions, avg_error

    def train_hetero_model(
        self, hidden_channels=64, heads=3, lr=0.005, weight_decay=5e-4
    ):
        """
        训练异构图神经网络模型

        参数:
            hidden_channels: 隐藏层神经元数量
            heads: 注意力头数量
            lr: 学习率
            weight_decay: 权重衰减

        返回:
            验证集的平均误差和损失
        """
        # 首先创建训练和验证掩码
        train_mask, val_mask, test_mask = create_data_masks(
            len(self.features_norm), self.config, self.device
        )

        # 标准化天线位置
        antenna_locations_norm = to_device(
            torch.tensor(
                self.labels_scaler.transform(self.antenna_locations.cpu().numpy()),
                dtype=torch.float32
            ), self.device
        )

        # 调用heterogeneous模块中的训练函数
        self.config['RFID_INSTANCE'] = self  # 添加实例的引用
        best_val_avg_distance, best_val_loss, self.hetero_model, train_losses, val_losses = train_hetero_model_func(
            self.features_norm,
            self.labels_norm,
            antenna_locations_norm,
            train_mask,
            val_mask,
            test_mask,
            self.labels_scaler,
            self.device,
            self.config,
            hidden_channels=hidden_channels,
            heads=heads,
            lr=lr,
            weight_decay=weight_decay
        )

        # 保存损失记录
        self.hetero_train_losses = train_losses
        self.hetero_val_losses = val_losses

        return best_val_avg_distance, best_val_loss

    def evaluate_prediction_hetero_accuracy(self, test_data=None, num_samples=50):
        """
        评估异构图模型在新标签预测上的准确性

        参数:
            test_data: 测试数据，格式为(特征，标签)的元组
            num_samples: 如果没有提供测试数据，随机抽样的样本数量

        返回:
            平均预测误差距离
        """
        if self.hetero_model is None:
            raise ValueError("异构图模型未训练。请先调用train_hetero_model。")

        if test_data is None:
            raise ValueError("必须提供 test_data 参数，不能为 None")
        else:
            test_features, test_labels = test_data
            test_features = to_device(
                torch.tensor(test_features, dtype=torch.float32), self.device
            )
            test_labels = to_device(
                torch.tensor(test_labels, dtype=torch.float32), self.device
            )

        # 标准化测试特征
        rssi_values = test_features[:, :4]
        phase_values = test_features[:, 4:8]
        rssi_norm = self.scaler_rssi.transform(rssi_values.cpu().numpy())
        phase_norm = self.scaler_phase.transform(phase_values.cpu().numpy())
        features_norm = np.hstack([rssi_norm, phase_norm])
        features_norm = to_device(
            torch.tensor(features_norm, dtype=torch.float32), self.device
        )

        # 预测结果列表
        predicted_positions = []

        # 确保MLP模型已训练，用于初始估计
        if self.mlp_model is None:
            print("mlp_model为空")
            self.train_mlp_model()

        # 标准化天线位置
        antenna_locations_norm = to_device(
            torch.tensor(
                self.labels_scaler.transform(self.antenna_locations.cpu().numpy()),
                dtype=torch.float32
            ), self.device
        )

        # 单独预测每个测试样本
        for i in range(len(features_norm)):
            # 提取单个样本特征
            sample_features = features_norm[i:i + 1]

            # 使用MLP估计初始位置
            self.mlp_model.eval()
            with torch.no_grad():
                mlp_pred = self.mlp_model(sample_features)
            temp_labels = to_device(mlp_pred, self.device)

            # 使用add_new_node_to_hetero_graph添加新节点
            # 首先创建初始异构图
            full_hetero_data = create_heterogeneous_graph_data(
                self.features_norm,
                self.labels_norm,
                antenna_locations_norm,
                k=self.config['K'],
                device=self.device
            )

            # 添加新节点
            new_hetero_data = add_new_node_to_hetero_graph(
                full_hetero_data,
                sample_features,
                temp_labels,
                k=self.config['K'],
                device=self.device
            )

            # 准备edge_index和edge_attr字典
            edge_index_dict = {
                ('tag', 'to', 'tag'): new_hetero_data['tag', 'to', 'tag'].edge_index,
                ('tag', 'to', 'antenna'):
                    new_hetero_data['tag', 'to', 'antenna'].edge_index,
                ('antenna', 'to', 'tag'):
                    new_hetero_data['antenna', 'to', 'tag'].edge_index
            }

            edge_attr_dict = {
                ('tag', 'to', 'tag'): new_hetero_data['tag', 'to', 'tag'].edge_attr,
                ('tag', 'to', 'antenna'):
                    new_hetero_data['tag', 'to', 'antenna'].edge_attr,
                ('antenna', 'to', 'tag'):
                    new_hetero_data['antenna', 'to', 'tag'].edge_attr
            }

            # 准备节点特征字典
            x_dict = {
                'tag': new_hetero_data['tag'].x,
                'antenna': new_hetero_data['antenna'].x
            }

            # 使用异构图模型进行预测
            self.hetero_model.eval()
            with torch.no_grad():
                out = self.hetero_model(x_dict, edge_index_dict, edge_attr_dict)

                # 获取新节点的预测 - 使用tag_mask找到新节点
                new_node_idx = torch.where(new_hetero_data['tag'].tag_mask)[0]
                pred_pos = out[new_node_idx]
                predicted_positions.append(pred_pos)

        # 堆叠预测结果
        predicted_positions = torch.cat(predicted_positions, dim=0)

        # 反标准化以获得实际坐标
        predicted_positions_orig = self.labels_scaler.inverse_transform(
            predicted_positions.cpu().numpy()
        )
        predicted_positions_orig = to_device(
            torch.tensor(predicted_positions_orig, dtype=torch.float32), self.device
        )

        # 计算欧几里得距离误差
        distances = torch.sqrt(
            torch.sum((test_labels - predicted_positions_orig)**2, dim=1)
        )
        avg_distance = torch.mean(distances).item()

        if self.config['PREDICTION_LOG']:
            log_message = "\n异构图模型新标签位置预测评估:\n"
            log_message += f"测试样本数量: {len(test_features)}\n"
            log_message += f"平均预测误差: {avg_distance:.2f}米\n"
            log_message += f"最大误差: {torch.max(distances).item():.2f}米\n"
            log_message += f"最小误差: {torch.min(distances).item():.2f}米\n"
            log_message += f"误差标准差: {torch.std(distances).item():.2f}米\n"

            # 计算不同误差阈值下的准确率
            for threshold in [0.5, 1.0, 1.5, 2.0]:
                accuracy = (distances < threshold).float().mean().item() * 100
                log_message += f"误差 < {threshold}米的准确率: {accuracy:.2f}%\n"

            # 打印到控制台并写入日志
            self.prediction_logger.info(log_message)

        return avg_distance

    def plot_all_models_loss_comparison(self, save_path=None):
        """
        绘制MLP、GAT和异构图模型的训练和验证损失对比图

        参数:
            save_path: 图像保存路径，不提供则显示图像
        """
        # 检查是否有足够的损失数据
        has_mlp = len(self.mlp_train_losses) > 0
        has_gat = len(self.gat_train_losses) > 0
        has_hetero = len(self.hetero_train_losses) > 0

        if not (has_mlp or has_gat or has_hetero):
            print("没有可用的损失数据来绘制图表")
            return

        plt.figure(figsize=(12, 8))

        # 确保目录存在
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 绘制训练损失
        plt.subplot(2, 1, 1)
        max_epoch = 50
        if has_mlp:
            epochs_mlp = range(1, min(len(self.mlp_train_losses), max_epoch) + 1)
            plt.plot(
                epochs_mlp, self.mlp_train_losses[:max_epoch], 'b-', label='MLP训练损失'
            )
        if has_gat:
            epochs_gat = range(1, min(len(self.gat_train_losses), max_epoch) + 1)
            plt.plot(
                epochs_gat, self.gat_train_losses[:max_epoch], 'r-', label='GAT训练损失'
            )
        if has_hetero:
            epochs_hetero = range(1, min(len(self.hetero_train_losses), max_epoch) + 1)
            plt.plot(
                epochs_hetero,
                self.hetero_train_losses[:max_epoch],
                'g-',
                label='异构图训练损失'
            )

        plt.title('不同模型训练损失对比')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True)

        # 绘制验证损失
        plt.subplot(2, 1, 2)
        if has_mlp:
            plt.plot(epochs_mlp, self.mlp_val_losses[:max_epoch], 'b-', label='MLP验证损失')
        if has_gat:
            plt.plot(epochs_gat, self.gat_val_losses[:max_epoch], 'r-', label='GAT验证损失')
        if has_hetero:
            plt.plot(
                epochs_hetero,
                self.hetero_val_losses[:max_epoch],
                'g-',
                label='异构图验证损失'
            )

        plt.title('不同模型验证损失对比')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


def main():
    """运行实验的主函数"""
    # 加载测试数据
    test_features, test_labels, test_features_np, test_labels_np, _ = load_and_preprocess_test_data(
        CONFIG['TEST_DATA_PATH']
    )

    # 检查是否启用网格搜索
    if CONFIG['GRID_SEARCH']:
        # 运行网格搜索
        print("执行超参数网格搜索")

        # 使用配置文件中的网格搜索参数
        best_models_results, sorted_model_names = run_grid_search(
            CONFIG,
            RFIDLocalization,
            k_range=CONFIG['GRID_SEARCH_PARAMS']['k_range'],
            lr_range=CONFIG['GRID_SEARCH_PARAMS']['lr_range'],
            weight_decay_range=CONFIG['GRID_SEARCH_PARAMS']['weight_decay_range'],
            hidden_channels_range=CONFIG['GRID_SEARCH_PARAMS']['hidden_channels_range'],
            heads_range=CONFIG['GRID_SEARCH_PARAMS']['heads_range'],
            quick_search=CONFIG['QUICK_SEARCH'],
            dropout_range=CONFIG['GRID_SEARCH_PARAMS']['dropout_range']
        )

        # 获取最佳模型类型
        best_model_type = sorted_model_names[0]  # 最佳模型名称
        print(f"网格搜索确定的最佳模型类型: {best_model_type}")
        print(f"所有模型按性能排序: {', '.join(sorted_model_names)}")
    else:
        # 不进行网格搜索，使用默认参数
        print("使用默认超参数训练模型")
        best_models_results = CONFIG['MODEL_PARAMS']
        sorted_model_names = ["GAT", "MLP", "异构图"]  # 默认顺序
        config = CONFIG.copy()

    # 确保结果目录存在
    os.makedirs(CONFIG['RESULTS_DIR'], exist_ok=True)

    # 初始化一个共享的定位系统
    # 创建一个基础配置
    base_config = CONFIG.copy()
    # 使用共享的定位系统实例来训练所有模型
    shared_localization = RFIDLocalization(base_config)

    # 存储错误信息的字典
    errors = {}

    # 对每个模型分别训练
    for model_name in ["MLP", "GAT", "异构图"]:
        if model_name in best_models_results:
            print(f"\n=================== 训练{model_name}模型 ===================")
            # 每个模型使用自己的最佳参数
            model_params = best_models_results[model_name]

            # 更新配置中的K参数
            if 'K' in model_params:
                shared_localization.config['K'] = model_params['K']

            # 根据不同模型类型训练
            if model_name == "MLP" and shared_localization.config['OPEN_MLP']:
                print("训练MLP参数：", model_params)

                best_val_loss, best_val_avg_distance, best_model = shared_localization.train_mlp_model(
                    hidden_channels=model_params.get('hidden_channels', 128),
                    dropout=model_params.get('dropout', 0.1),
                    lr=model_params.get('lr', 0.001),
                    weight_decay=model_params.get('weight_decay', 0.0005)
                )
                # 在新数据上评估MLP模型
                mlp_prediction_error = shared_localization.evaluate_mlp_on_new_data(
                    test_features_np, test_labels_np
                )
                errors['MLP'] = mlp_prediction_error
                print(f"MLP模型在新标签上的平均预测误差: {mlp_prediction_error:.2f}米")

            elif model_name == "GAT" and shared_localization.config['OPEN_GAT']:
                # 创建GAT模型
                shared_localization.model = GATLocalizationModel(
                    in_channels=shared_localization.features_norm.shape[1],
                    hidden_channels=model_params.get('hidden_channels', 64),
                    out_channels=2,
                    heads=model_params.get('heads', 3)
                )

                # 训练GAT模型
                best_val_avg_distance, best_val_loss = shared_localization.train_gat_model(
                    hidden_channels=model_params.get('hidden_channels', 64),
                    heads=model_params.get('heads', 3),
                    lr=model_params.get('lr', 0.005),
                    weight_decay=model_params.get('weight_decay', 5e-4)
                )

                # 评估GAT模型
                np.random.seed(shared_localization.config['RANDOM_SEED'])
                gat_prediction_error = shared_localization.evaluate_prediction_GAT_accuracy(
                    test_data=(test_features_np, test_labels_np),
                    num_samples=len(test_features_np)
                )
                errors['GAT'] = gat_prediction_error
                print(f"GAT模型在新标签上的平均预测误差: {gat_prediction_error:.2f}米")

            elif model_name == "异构图" and shared_localization.config['OPEN_HETERO']:
                # 训练异构图模型
                best_val_avg_distance, best_val_loss = shared_localization.train_hetero_model(
                    hidden_channels=model_params.get('hidden_channels', 64),
                    heads=model_params.get('heads', 3),
                    lr=model_params.get('lr', 0.005),
                    weight_decay=model_params.get('weight_decay', 5e-4)
                )

                # 评估异构图模型
                np.random.seed(shared_localization.config['RANDOM_SEED'])
                hetero_prediction_error = shared_localization.evaluate_prediction_hetero_accuracy(
                    test_data=(test_features_np, test_labels_np),
                    num_samples=len(test_features_np)
                )
                errors['异构图'] = hetero_prediction_error
                print(f"异构图模型在新标签上的平均预测误差: {hetero_prediction_error:.2f}米")

    # KNN和LANDMARC模型评估
    # 直接使用shared_localization进行评估
    if shared_localization.config['OPEN_KNN']:
        # 训练和评估KNN模型
        features_array = shared_localization.features.cpu().numpy()
        labels_array = shared_localization.labels.cpu().numpy()

        # 直接调用导入的knn_localization函数
        knn_model = knn_localization(
            features_array, labels_array, n_neighbors=shared_localization.config['K']
        )

        # 直接调用导入的evaluate_knn_on_test_set函数
        avg_error = evaluate_knn_on_test_set(
            knn_model, test_features_np, test_labels_np
        )
        errors['KNN'] = avg_error
        print(f"KNN模型测试集平均误差: {avg_error:.2f}米")

        # 记录详细的KNN评估日志
        if shared_localization.config['PREDICTION_LOG']:
            # 预测测试集
            predictions = knn_model.predict(test_features_np)
            # 计算实际米数的误差
            distances = np.sqrt(np.sum((test_labels_np - predictions)**2, axis=1))

            log_message = "\nKNN模型位置预测评估:\n"
            log_message += f"测试样本数量: {len(test_features_np)}\n"
            log_message += f"平均预测误差: {avg_error:.2f}米\n"
            log_message += f"最大误差: {np.max(distances):.2f}米\n"
            log_message += f"最小误差: {np.min(distances):.2f}米\n"
            log_message += f"误差标准差: {np.std(distances):.2f}米\n"

            # 计算不同误差阈值下的准确率
            for threshold in [0.5, 1.0, 1.5, 2.0]:
                accuracy = np.mean(distances < threshold) * 100
                log_message += f"误差 < {threshold}米的准确率: {accuracy:.2f}%\n"

            # 打印到控制台并写入日志
            shared_localization.prediction_logger.info(log_message)

    if shared_localization.config['OPEN_LANDMARC']:
        # 对测试集使用LANDMARC算法
        # 提取参考标签的特征和位置
        reference_features = shared_localization.features
        reference_locations = shared_localization.labels

        # 直接调用导入的landmarc_localization函数
        _, landmarc_error = landmarc_localization(
            reference_features,
            reference_locations,
            test_features,
            test_labels,
            k=shared_localization.config['K']
        )
        errors['LANDMARC'] = landmarc_error
        print(f"LANDMARC算法测试集平均误差: {landmarc_error:.2f}米")

    # 打印所有模型的性能比较
    if errors:
        print("\n======= 所有模型性能比较 =======")
        # 按照误差从小到大排序
        sorted_errors = sorted(errors.items(), key=lambda x: x[1])
        for i, (model, error) in enumerate(sorted_errors):
            print(f"{i+1}. {model}: {error:.2f}米")

        # 找出性能最好的模型
        best_model = sorted_errors[0][0]
        print(f"\n实际测试中性能最佳的模型是: {best_model}")

        # 绘制损失对比图
        shared_localization.plot_all_models_loss_comparison(
            save_path=CONFIG['MODEL_COMPARISON_IMAGE']
        )

        # 如果网格搜索确定了最佳模型类型，与实际测试结果进行比较
        if CONFIG['GRID_SEARCH'] and sorted_model_names:
            if sorted_model_names[0] == best_model:
                print(f"网格搜索结果与实际测试结果一致，都为{best_model}模型")
            else:
                print(f"网格搜索结果({sorted_model_names[0]})与实际测试结果({best_model})不一致")


if __name__ == "__main__":
    main()
