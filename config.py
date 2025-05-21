# -*- coding: utf-8 -*-
"""
RFID定位系统配置文件
"""

# 基本配置
CONFIG = {
    'K': 7,  # KNN的K值
    'RANDOM_SEED': 32,  # 随机种子，改为更稳定的值
    'OPEN_KNN': True,  # 启用KNN算法进行比较
    'TRAIN_LOG': False,  # 启用训练日志
    'PREDICTION_LOG': False,  # 启用预测日志
    'GRID_SEARCH': True,  # 是否启用网格搜索
    'QUICK_SEARCH': False,  # 是否使用快速搜索（减少组合数量）
    'OPEN_MLP': True,  # 启用MLP算法进行比较
    'OPEN_GAT': True,  # 启用GAT算法进行比较
    'OPEN_HETERO': True,  # 启用异构图算法进行比较
    'DATA_AUGMENTATION': False,  # 是否启用数据增强
    'OPEN_LANDMARC': True,  # 启用LANDMARC算法进行比较

    # 天线位置坐标
    'ANTENNA_LOCATIONS': [
        [0.0, 0.0],  # 天线1 (0,0)
        [0.0, 10.0],  # 天线2 (0,10)
        [10.0, 0.0],  # 天线3 (10,0)
        [10.0, 10.0]  # 天线4 (10,10)
    ],

    # 数据文件路径
    'REFERENCE_DATA_PATH': 'data/rfid_reference_tags.csv',
    'TEST_DATA_PATH': 'data/rfid_test_tags.csv',

    # 模型默认参数
    'MODEL_PARAMS': {
        'GAT': {
            'K': 7,
            'lr': 0.001,
            'weight_decay': 1e-05,
            'hidden_channels': 32,
            'heads': 1
        },
        'MLP': {
            'lr': 0.005,
            'weight_decay': 5e-05,
            'hidden_channels': 64,
            'dropout': 0.1
        },
        '异构图': {
            'K': 8,
            'lr': 0.001,
            'weight_decay': 5e-05,
            'hidden_channels': 128,
            'heads': 2
        }
    },

    # 训练参数
    'EPOCHS': 1000,
    'PATIENCE': 50,  # 早停耐心值

    # 网格搜索参数范围
    'GRID_SEARCH_PARAMS': {
        'k_range': [5, 6, 7, 8],
        'lr_range': [0.001, 0.005, 0.01],
        'weight_decay_range': [1e-5, 5e-5, 1e-4, 5e-4],
        'hidden_channels_range': [32, 64, 128],
        'heads_range': [1, 2, 3],
        'dropout_range': [0.1, 0.2, 0.3]
    },

    # 结果保存路径
    'RESULTS_DIR': 'results',
    'MODEL_COMPARISON_IMAGE': 'results/all_models_loss_comparison.png',

    # 边属性权重配置
    'EDGE_ATTR_WEIGHTS': {
        'w1': 0.6,  # 距离影响权重增加
        'w2': 0.4,  # RSSI影响权重增加
        'w3': 0.3,  # 材质和大小影响权重降低，减少椅子材质的影响
    }
}
