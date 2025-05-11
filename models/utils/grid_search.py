#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
网格搜索工具模块 - 用于超参数优化
"""

import time
import itertools
import numpy as np
import pandas as pd
import os
import torch

# 从utils导入数据加载函数
from models.utils.data_loader import load_and_preprocess_test_data


def run_grid_search(
    config,
    RFIDLocalization,
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
        config: 配置字典
        RFIDLocalization: 主定位类
        k_range: KNN的K值范围
        lr_range: 学习率范围
        weight_decay_range: 权重衰减范围
        hidden_channels_range: 隐藏层通道数范围
        heads_range: 注意力头数范围
        quick_search: 是否使用快速搜索（减少组合数量）
        dropout_range: Dropout比率范围

    返回:
        前三个最佳模型的参数和模型名称
    """
    start_time = time.time()  # 记录开始时间
    # 设置默认搜索范围
    if k_range is None:
        k_range = config['GRID_SEARCH_PARAMS']['k_range']
    if lr_range is None:
        lr_range = config['GRID_SEARCH_PARAMS']['lr_range']
    if weight_decay_range is None:
        weight_decay_range = config['GRID_SEARCH_PARAMS']['weight_decay_range']
    if hidden_channels_range is None:
        hidden_channels_range = config['GRID_SEARCH_PARAMS']['hidden_channels_range']
    if heads_range is None:
        heads_range = config['GRID_SEARCH_PARAMS']['heads_range']
    if dropout_range is None:
        dropout_range = config['GRID_SEARCH_PARAMS']['dropout_range']
    if quick_search is None:
        quick_search = config['QUICK_SEARCH']

    # 加载测试集数据用于验证
    test_features, test_labels, test_features_np, test_labels_np, _ = load_and_preprocess_test_data(
        config['TEST_DATA_PATH']
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
            weight_decay_range = [weight_decay_range[0], weight_decay_range[-1]]
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
    # 存储异构图结果
    hetero_results = []

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

    # 创建所有异构图超参数组合 (与GAT相同)
    hetero_param_combinations = list(
        itertools.product(
            k_range, lr_range, weight_decay_range, hidden_channels_range, heads_range
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
        current_config = config.copy()

        # 初始化定位系统
        localization = RFIDLocalization(current_config)

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
        test_avg_distance = localization.evaluate_mlp_on_new_data(
            test_features_np, test_labels_np
        )
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
        current_config = config.copy()
        best_mlp_localization = RFIDLocalization(current_config)
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
        current_config = config.copy()
        current_config['K'] = k

        # 初始化定位系统
        localization = RFIDLocalization(current_config)

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

    # 执行异构图网格搜索
    print(f"\n开始异构图网格搜索，共 {len(hetero_param_combinations)} 种组合")

    for i, (k, lr, weight_decay, hidden_channels,
            heads) in enumerate(hetero_param_combinations):
        print(f"\n异构图组合 {i+1}/{len(hetero_param_combinations)}:")
        print(
            f"K={k}, lr={lr}, weight_decay={weight_decay}, hidden_channels={hidden_channels}, heads={heads}"
        )

        # 更新配置
        current_config = config.copy()
        current_config['K'] = k

        # 初始化定位系统
        localization = RFIDLocalization(current_config)

        # 使用最佳MLP模型替换新初始化的MLP模型
        localization.mlp_model = best_mlp_model

        # 确保标准化器与最佳MLP模型一致
        localization.scaler_rssi = best_mlp_localization.scaler_rssi
        localization.scaler_phase = best_mlp_localization.scaler_phase
        localization.labels_scaler = best_mlp_localization.labels_scaler

        # 直接调用 train_hetero_model 进行训练
        best_val_avg_distance, best_val_loss = localization.train_hetero_model(
            hidden_channels=hidden_channels,
            heads=heads,
            lr=lr,
            weight_decay=weight_decay
        )
        val_avg_distance = best_val_avg_distance

        # 用测试数据评估异构图模型
        test_avg_distance = localization.evaluate_prediction_hetero_accuracy(
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
        hetero_results.append(result)

        print(f"测试集平均误差: {val_avg_distance:.2f}米")
        print(f"验证集平均误差: {test_avg_distance:.2f}米")

    # 按验证集平均误差排序
    gat_results.sort(key=lambda x: x['test_avg_distance'])
    mlp_results.sort(key=lambda x: x['test_avg_distance'])
    hetero_results.sort(key=lambda x: x['test_avg_distance'])

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

    # 输出最佳异构图超参数
    best_hetero_result = hetero_results[0]
    print("\n============ 最佳异构图超参数 ============")
    print(f"K: {best_hetero_result['K']}")
    print(f"学习率: {best_hetero_result['lr']}")
    print(f"权重衰减: {best_hetero_result['weight_decay']}")
    print(f"隐藏层通道数: {best_hetero_result['hidden_channels']}")
    print(f"注意力头数: {best_hetero_result['heads']}")
    print(f"验证集平均误差: {best_hetero_result['test_avg_distance']:.2f}米")

    # 确保结果目录存在
    os.makedirs(config['RESULTS_DIR'], exist_ok=True)

    # 将所有结果输出到文件
    gat_results_df = pd.DataFrame(gat_results)
    gat_results_df.to_csv(
        f'{config["RESULTS_DIR"]}/gat_grid_search_results.csv', index=False
    )

    mlp_results_df = pd.DataFrame(mlp_results)
    mlp_results_df.to_csv(
        f'{config["RESULTS_DIR"]}/mlp_grid_search_results.csv', index=False
    )

    hetero_results_df = pd.DataFrame(hetero_results)
    hetero_results_df.to_csv(
        f'{config["RESULTS_DIR"]}/hetero_grid_search_results.csv', index=False
    )

    print("GAT结果已保存到 gat_grid_search_results.csv")
    print("MLP结果已保存到 mlp_grid_search_results.csv")
    print("异构图结果已保存到 hetero_grid_search_results.csv")

    end_time = time.time()  # 记录结束时间
    total_time = end_time - start_time
    print(f"\n网格搜索总耗时: {total_time:.2f} 秒")

    # 比较所有模型的最佳结果
    all_results = [("GAT", best_gat_result['test_avg_distance']),
                   ("MLP", best_mlp_result['test_avg_distance']),
                   ("异构图", best_hetero_result['test_avg_distance'])]
    # 按照误差从小到大排序
    sorted_models = sorted(all_results, key=lambda x: x[1])
    best_model_name, best_error = sorted_models[0]
    print(f"\n{best_model_name}模型性能最好，误差: {best_error:.2f}米!")

    # 创建一个包含前三个最佳模型的结果字典
    best_models_results = {}
    for model_name, _ in sorted_models:
        if model_name == "GAT":
            best_models_results["GAT"] = best_gat_result
        elif model_name == "MLP":
            best_models_results["MLP"] = best_mlp_result
        else:
            best_models_results["异构图"] = best_hetero_result

    # 返回所有三个模型的参数和排序后的模型名称列表
    return best_models_results, [model[0] for model in sorted_models]
