import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

from .model import GATLocalizationModel


def train_gat_model(
    localization, hidden_channels=64, heads=3, lr=0.005, weight_decay=5e-4
):
    """训练GAT模型。localization为RFIDLocalization实例。"""
    full_graph_data = create_graph_data(
        localization,
        localization.features_norm,
        localization.labels_norm,
        k=localization.config['K']
    )
    train_mask, val_mask, test_mask = create_data_masks(
        len(localization.features_norm), localization.config, localization.device
    )
    full_graph_data.train_mask = train_mask
    full_graph_data.val_mask = val_mask
    full_graph_data = to_device(full_graph_data, localization.device)
    torch.manual_seed(localization.config['RANDOM_SEED'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(localization.config['RANDOM_SEED'])
        torch.cuda.manual_seed_all(localization.config['RANDOM_SEED'])
    localization.model = GATLocalizationModel(
        in_channels=full_graph_data.x.shape[1],
        hidden_channels=hidden_channels,
        out_channels=2,
        heads=heads
    ).to(localization.device)
    data_min = torch.as_tensor(
        localization.labels_scaler.data_min_, dtype=torch.float32
    ).to(localization.device)
    data_range = torch.as_tensor(
        localization.labels_scaler.data_range_, dtype=torch.float32
    ).to(localization.device)
    torch.manual_seed(localization.config['RANDOM_SEED'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(localization.config['RANDOM_SEED'])
        torch.cuda.manual_seed_all(localization.config['RANDOM_SEED'])
    optimizer = torch.optim.Adam(
        localization.model.parameters(), lr=lr, weight_decay=weight_decay
    )
    loss_fn = torch.nn.MSELoss()
    best_val_loss = float('inf')
    best_model = None
    patience = localization.config.get('PATIENCE', 50)
    counter = 0
    best_val_avg_distance = float('inf')
    localization.gat_train_losses = []
    localization.gat_val_losses = []
    for epoch in range(localization.config.get('EPOCHS', 1000)):
        localization.model.train()
        optimizer.zero_grad()
        out = localization.model(full_graph_data)
        train_loss = loss_fn(
            out[full_graph_data.train_mask],
            full_graph_data.y[full_graph_data.train_mask]
        )
        train_loss.backward()
        optimizer.step()
        localization.model.eval()
        with torch.no_grad():
            val_loss = loss_fn(
                out[full_graph_data.val_mask],
                full_graph_data.y[full_graph_data.val_mask]
            )
            out_orig = out * data_range + data_min
            y_orig = full_graph_data.y * data_range + data_min
            train_mask_orig = full_graph_data.train_mask
            val_mask_orig = full_graph_data.val_mask
            if train_mask_orig.sum() > 0:
                train_distances = torch.sqrt(
                    torch.sum((out_orig[train_mask_orig] - y_orig[train_mask_orig])**2,
                              dim=1)
                )
                train_accuracy = (train_distances < 0.3).float().mean().item() * 100
                train_avg_distance = train_distances.mean().item()
            else:
                train_accuracy = 0
                train_avg_distance = 0
            if val_mask_orig.sum() > 0:
                val_distances = torch.sqrt(
                    torch.sum((out_orig[val_mask_orig] - y_orig[val_mask_orig])**2,
                              dim=1)
                )
                val_accuracy = (val_distances < 0.3).float().mean().item() * 100
                val_avg_distance = val_distances.mean().item()
            else:
                val_accuracy = 0
                val_avg_distance = float('inf')
        localization.gat_train_losses.append(train_loss.item())
        localization.gat_val_losses.append(val_loss.item())
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_avg_distance = val_avg_distance
            best_model = localization.model.state_dict().copy()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                if localization.config['TRAIN_LOG']:
                    print(
                        f"轮次 {epoch}\n训练集 - 损失: {train_loss.item():.4f}, 准确率: {train_accuracy:.2f}%, 平均误差: {train_avg_distance:.2f}米\n验证集 - 损失: {val_loss.item():.4f}, 准确率: {val_accuracy:.2f}%, 平均误差: {val_avg_distance:.2f}米"
                    )
                    print(f"\n触发早停！在轮次 {epoch} 停止训练")
                    print(f"最佳验证损失: {best_val_loss:.4f}")
                localization.model.load_state_dict(best_model)
                break
        if epoch % 100 == 0 and localization.config['TRAIN_LOG']:
            print(
                f"轮次 {epoch}\n训练集 - 损失: {train_loss.item():.4f}, 准确率: {train_accuracy:.2f}%, 平均误差: {train_avg_distance:.2f}米\n验证集 - 损失: {val_loss.item():.4f}, 准确率: {val_accuracy:.2f}%, 平均误差: {val_avg_distance:.2f}米"
            )
    return best_val_avg_distance, best_val_loss.item()


def evaluate_prediction_GAT_accuracy(localization, test_data=None, num_samples=50):
    """评估GAT模型在新标签预测上的准确性。localization为RFIDLocalization实例。"""
    if localization.model is None:
        raise ValueError("模型未训练。请先调用train_gat_model。")
    if test_data is None:
        raise ValueError("必须提供 test_data 参数，不能为 None")
    test_features, test_labels = test_data
    test_features = to_device(
        torch.tensor(test_features, dtype=torch.float32), localization.device
    )
    test_labels = to_device(
        torch.tensor(test_labels, dtype=torch.float32), localization.device
    )
    rssi_values = test_features[:, :4]
    phase_values = test_features[:, 4:8]
    rssi_norm = localization.scaler_rssi.transform(rssi_values.cpu().numpy())
    phase_norm = localization.scaler_phase.transform(phase_values.cpu().numpy())
    features_new = to_device(
        torch.tensor(np.hstack([rssi_norm, phase_norm]), dtype=torch.float32),
        localization.device
    )
    predicted_positions = []
    if localization.mlp_model is None:
        localization.train_mlp_model()
    for i in range(len(features_new)):
        sample_features = features_new[i:i + 1]
        localization.mlp_model.eval()
        with torch.no_grad():
            mlp_pred = localization.mlp_model(sample_features)
        temp_labels = to_device(mlp_pred, localization.device)
        all_features = torch.cat([localization.features_norm, sample_features], dim=0)
        all_labels = torch.cat([localization.labels_norm, temp_labels], dim=0)
        graph_data = create_graph_data(
            localization,
            all_features,
            all_labels,
            k=localization.config['K'],
            pos=localization.labels
        )
        localization.model.eval()
        with torch.no_grad():
            out = localization.model(graph_data)
            pred_idx = len(localization.features_norm)
            pred_pos = out[pred_idx]
            predicted_positions.append(pred_pos)
    predicted_positions = torch.stack(predicted_positions)
    predicted_positions_orig = localization.labels_scaler.inverse_transform(
        predicted_positions.cpu().numpy()
    )
    predicted_positions_orig = to_device(
        torch.tensor(predicted_positions_orig, dtype=torch.float32), localization.device
    )
    distances = torch.sqrt(
        torch.sum((test_labels - predicted_positions_orig)**2, dim=1)
    )
    avg_distance = torch.mean(distances).item()
    return avg_distance


def create_graph_data(localization, features_norm, labels_norm, k=None, pos=None):
    """为GAT模型创建图数据"""
    if k is None:
        k = localization.config['K']
    if pos is None:
        pos = localization.labels
    from sklearn.neighbors import kneighbors_graph
    import torch
    import numpy as np
    from torch_geometric.data import Data
    adj_matrix = kneighbors_graph(
        torch.as_tensor(
            np.hstack([
                0.2 * features_norm.cpu().numpy(), 0.8 * labels_norm.cpu().numpy()
            ])
        ),
        n_neighbors=k,
        mode='distance',
    )
    adj_matrix_dense = torch.as_tensor(adj_matrix.toarray(), dtype=torch.float32)
    edge_index = to_device(
        torch.nonzero(adj_matrix_dense, as_tuple=False).t(), localization.device
    )
    adj_matrix_coo = adj_matrix.tocoo()
    edge_attr = to_device(
        torch.tensor(adj_matrix_coo.data, dtype=torch.float32), localization.device
    )
    return Data(
        x=to_device(features_norm, localization.device),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=to_device(labels_norm, localization.device),
        pos=pos
    )


def to_device(data, device):
    """将数据移动到指定设备"""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    return data


def create_data_masks(num_nodes, config, device, test_size=0.2, val_size=0.2):
    """创建训练、验证和测试集的掩码"""
    import numpy as np
    from sklearn.model_selection import train_test_split
    import torch
    indices = np.arange(num_nodes)
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=config['RANDOM_SEED']
    )
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=val_size / (1 - test_size),
        random_state=config['RANDOM_SEED']
    )
    train_mask = to_device(torch.zeros(num_nodes, dtype=torch.bool), device)
    val_mask = to_device(torch.zeros(num_nodes, dtype=torch.bool), device)
    test_mask = to_device(torch.zeros(num_nodes, dtype=torch.bool), device)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask
