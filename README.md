# RFID-GATv2：基于图注意力网络的 RFID 定位系统

## 项目简介

本项目实现了一套基于图注意力网络（GATv2）的 RFID 定位方法，并集成了 MLP、KNN、LANDMARC 等多种定位算法，支持超参数网格搜索和多模型对比。项目适用于 RFID 标签定位的科研与工程应用。

## 主要功能

- 基于 GATv2 的 RFID 标签定位
- 支持 MLP、KNN、LANDMARC 等传统与深度学习方法对比
- 支持超参数网格搜索与快速搜索
- 支持数据标准化、数据增强
- 结果可视化与误差分析

## 依赖环境

- Python 3.7+
- numpy
- pandas
- torch
- torch_geometric
- scikit-learn
- matplotlib
- networkx

如未安装依赖，请先运行：

```bash
pip install numpy pandas torch torch_geometric scikit-learn matplotlib networkx
```

## 数据说明

- `data/rfid_reference_tags.csv`：参考标签数据，包含 RSSI、相位、位置等字段。
- `data/rfid_test_tags.csv`：测试标签数据，结构同上。

主要字段：

- rssi_antenna1~4：四根天线的 RSSI 值
- phase_antenna1~4：四根天线的相位值
- true_x, true_y：标签真实坐标

## 目录结构

```
├── test.py                  # 主程序，包含全部模型与实验流程
├── get_min.py               # 简单脚本，用于获取最小测试误差
├── data/                    # 存放原始数据集
│   ├── rfid_reference_tags.csv
│   └── rfid_test_tags.csv
├── results/                 # 存放实验结果
│   ├── mlp_grid_search_results.csv
│   └── gat_grid_search_results.csv
├── images/                  # 可视化图片（如热力图、布局图等）
├── .style.yapf              # 代码风格配置
```

## 运行方法

1. 准备好数据集，放入`data/`目录。
2. 运行主程序：

```bash
python test.py
```

3. 可根据需要修改`CONFIG`字典，开启/关闭不同算法、日志、网格搜索等。

## 结果输出

- 主要实验结果（误差、最优参数等）会输出到控制台。
- 网格搜索结果会保存在`results/`目录下的 csv 文件中。
- 可视化图片保存在`images/`目录。
