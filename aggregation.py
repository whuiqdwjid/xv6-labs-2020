import numpy as np
import torch
import copy
from utils import cosine_similarity_score
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid
# 1、10%左右
# import torch
# import numpy as np
#
# def flatten_weights(weights_dict):
#     """将权重字典中的权重展平为一维张量"""
#     return torch.cat([w.view(-1) for w in weights_dict.values()])
#
# def unflatten_weights(flat_weights, reference_dict):
#     """将展平的权重张量恢复为原始字典结构"""
#     unflattened = {}
#     i = 0
#     for key, weight in reference_dict.items():
#         weight_size = weight.numel()
#         unflattened[key] = flat_weights[i:i + weight_size].view_as(weight)
#         i += weight_size
#     return unflattened
#
# def bulyan(local_weights, num_byzantine):
#     all_updates = torch.stack([flatten_weights(weights) for weights in local_weights])
#     nusers = len(local_weights)
#
#     # Bulyan 算法逻辑
#     bulyan_cluster = []
#     remaining_updates = all_updates
#     all_indices = np.arange(len(all_updates))
#
#     while len(bulyan_cluster) < (nusers - 2 * num_byzantine):
#         distances = []
#         for update in remaining_updates:
#             distance = torch.norm((remaining_updates - update), dim=1) ** 2
#             distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)
#
#         distances = torch.sort(distances, dim=1)[0]
#         scores = torch.sum(distances[:, :len(remaining_updates) - 2 - num_byzantine], dim=1)
#         indices = torch.argsort(scores)[:len(remaining_updates) - 2 - num_byzantine]
#
#         selected_index = indices[0].cpu().numpy()
#         bulyan_cluster.append(remaining_updates[selected_index])
#         remaining_updates = np.delete(remaining_updates, selected_index, axis=0)
#
#     bulyan_cluster = torch.stack(bulyan_cluster)
#
#     # 计算聚合后的权重
#     n, d = bulyan_cluster.shape
#     param_med = torch.median(bulyan_cluster, dim=0)[0]
#     sort_idx = torch.argsort(torch.abs(bulyan_cluster - param_med), dim=0)
#     sorted_params = bulyan_cluster[sort_idx, torch.arange(d)[None, :]]
#
#     aggregated_weights_flat = torch.mean(sorted_params[:n - 2 * num_byzantine], dim=0)
#
#     # 将展平的权重恢复为原始字典结构
#     global_weights = unflatten_weights(aggregated_weights_flat, local_weights[0])
#
#     return global_weights


# #2.bulyan
# import torch
#
# def compute_distance_score(weight, all_weights):
#     # This is a placeholder for the actual distance calculation
#     # You will need to implement the correct logic as per Krum algorithm
#     # For example, it could calculate the sum of Euclidean distances
#     # from this weight to all other weights
#     score = 0
#     for other_weight in all_weights:
#         if other_weight is not weight:
#             # Convert the model weights to vectors if they are not already
#             w = torch.nn.utils.parameters_to_vector(weight.values())
#             other_w = torch.nn.utils.parameters_to_vector(other_weight.values())
#             # Calculate the Euclidean distance and add it to the score
#             distance = torch.norm(w - other_w).item()
#             score += distance
#     return score
#
#
# def krum(weights):
#     # 计算所有权重之间的距离
#     n = len(weights)
#     distances = np.zeros((n, n))
#     for i in range(n):
#         for j in range(i + 1, n):
#             # 将模型权重转换为向量
#             w_i = torch.nn.utils.parameters_to_vector(weights[i].values())
#             w_j = torch.nn.utils.parameters_to_vector(weights[j].values())
#             # 计算欧几里得距离
#             distances[i, j] = distances[j, i] = torch.norm(w_i - w_j).item()
#     # 计算每个模型更新的得分（距离之和）
#     scores = [compute_distance_score(w, weights) for w in weights]
#     # 返回得分最小的模型的索引
#     return np.argmin(scores)
#     # # 选出得分最小的模型更新
#     # selected_index = np.argmin(scores)
#     # # 返回选出的模型权重
#     # return weights[selected_index]
#
#
# def bulyan(weights, num_byzantine):
#     num_models = len(weights)
#     # 初始化得分，需要一些适当的计算逻辑
#     scores = [compute_distance_score(w, weights) for w in weights]
#     # 使用Krum来选择一部分模型
#     krum_indices = set()
#     while len(krum_indices) < num_models - num_byzantine:
#         # 用得分而非权重来调用krum
#         idx = krum(scores)
#         krum_indices.add(idx)
#         # 将选中模型的得分设为无穷大以避免重新选择
#         scores[idx] = np.inf
#     # 提取Krum选出的模型权重
#     selected_weights = [weights[i] for i in krum_indices]
#     # 在这些权重上应用trimmed mean
#     aggregated_weights = {}
#     for key in selected_weights[0]:
#         layer_weights = np.array([w[key].numpy() for w in selected_weights])
#         aggregated_weights[key] = torch.from_numpy(trimmed_mean(layer_weights, num_byzantine))
#     return aggregated_weights

def trimmed_mean(weights, m):
    # Ensure there are enough elements to perform trimming
    if len(weights) > 2 * m:
        trimmed = weights[m:-m]
        if trimmed.size > 0:
            return np.mean(trimmed, axis=0)
        else:
            # Handle the case where the slice is empty
            return np.array([np.nan])  # Or handle it some other way
    else:
        # If there aren't enough elements, handle accordingly
        return np.array([np.nan])  # Or handle it some other way

def Bulyan_krum(weights):
    n = len(weights)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            w_i = torch.nn.utils.parameters_to_vector(weights[i].values())
            w_j = torch.nn.utils.parameters_to_vector(weights[j].values())
            distances[i, j] = distances[j, i] = torch.norm(w_i - w_j).item()
    # 直接基于距离矩阵计算得分，不使用compute_distance_score
    scores = distances.sum(axis=1)
    return np.argmin(scores)

def projection_weights(w,projection):
    w_projection = copy.deepcopy(w[0])
    for key in w_projection.keys():
        print(f'key {key} processing projection and add')
        for i in range(1, len(w)):
            #乘权重后加和
            print(f'processing ID {i} weight')
            w_projection[key] += w[i][key]*projection[i]
    return w_projection

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def krum(weights):
    # 计算所有权重之间的距离
    n = len(weights)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            # 将模型权重转换为向量
            w_i = torch.nn.utils.parameters_to_vector(weights[i].values())
            w_j = torch.nn.utils.parameters_to_vector(weights[j].values())
            # 计算欧几里得距离
            distances[i, j] = distances[j, i] = torch.norm(w_i - w_j).item()
    # 计算每个模型更新的得分（距离之和）
    scores = distances.sum(axis=0)
    # 选出得分最小的模型更新
    selected_index = np.argmin(scores)
    # 返回选出的模型权重
    return weights[selected_index]

def multi_krum(grads, k=2):
    """
    Multi-Krum algorithm for federated learning model aggregation.

    Parameters:
    - grads: List of gradients from different clients.
    - k: Number of gradients to consider in the Krum aggregation.

    Returns:
    - Aggregated gradient using Multi-Krum.
    """
    num_clients = len(grads)
    grad_norms = np.linalg.norm(grads, axis=1)  # Calculate the norm of each gradient

    # Calculate the pairwise distances between gradients
    distances = np.linalg.norm(grads[:, None, :] - grads[None, :, :], axis=-1)

    # Exclude oneself from consideration
    for i in range(num_clients):
        distances[i, i] = np.inf
    # Calculate the scores for each client
    scores = np.sum(np.sort(distances, axis=1)[:, :num_clients - k], axis=1)
    # Select the client with the minimum score
    selected_client = np.argmin(scores)
    # Aggregate the gradients
    aggregated_grad = grads[selected_client]
    return aggregated_grad


def median_aggregate(weights):
    # 假定weights是一个包含多个权重字典的列表
    aggregated_weights = {}
    for key in weights[0].keys():
        # 提取所有权重的相同参数
        param_values = torch.stack([w[key] for w in weights])
        # 计算中位数
        aggregated_weights[key] = torch.median(param_values, dim=0)[0]
    return aggregated_weights


def trimmed_mean_aggregate(weights, trim_ratio):
    """
    计算给定权重的 trimmed mean。
    :param weights: 包含多个客户端权重字典的列表。
    :param trim_ratio: 去除的比例（每一端）。
    :return: 聚合后的权重字典。
    """
    num_weights = len(weights)
    num_to_trim = int(trim_ratio * num_weights)

    aggregated_weights = {}
    for key in weights[0].keys():
        # 提取所有权重的相同参数
        param_values = torch.stack([w[key].data for w in weights], dim=0)
        # 计算中位数
        param_values, _ = torch.sort(param_values, dim=0)
        trimmed = param_values[num_to_trim:-num_to_trim] if num_to_trim > 0 else param_values
        aggregated_weights[key] = trimmed.mean(dim=0)
    return aggregated_weights


def bulyan(weights, num_byzantine):
    num_models = len(weights)
    krum_indices = set()
    while len(krum_indices) < num_models - num_byzantine:
        idx = Bulyan_krum(weights)  # 直接传递权重数组
        krum_indices.add(idx)

        # Update the weights list by excluding the selected index
        # This should be done carefully to avoid emptying the list prematurely
        if len(weights) > 1:  # Ensure there is more than one weight set before removing
            weights.pop(idx)  # This removes the weight at index 'idx'
        else:
            break  # If there's only one or no weight sets left, break the loop

    # After this point, 'weights' should not be empty if the loop condition is correct
    # Ensure 'selected_weights' contains the correct indices
    # 提取Krum选出的模型权重
    selected_weights = [weights[i] for i in krum_indices if i < len(weights)]
    # 在这些权重上应用trimmed mean
    aggregated_weights = {}
    for key in selected_weights[0]:
        layer_weights = np.array([w[key].numpy() for w in selected_weights])
        aggregated_weights[key] = torch.from_numpy(trimmed_mean(layer_weights, num_byzantine).astype(np.float32))
    return aggregated_weights


def fang_aggregated(local_weights, global_model, aggregation_methods, device, trim_ratio=0.1):
    """
    Fang-A防御聚合算法

    Args:
    - local_weights (list of dict): List of local model weights from each client.
    - global_model (torch.nn.Module): The global model for federated learning.
    - aggregation_methods (dict): A dictionary of available robust aggregation functions.
    - device (torch.device): The device on which to perform computations.
    - trim_ratio (float): The ratio of gradients to discard, defaults to 0.1 (10%).

    Returns:
    - dict: Aggregated global model weights.
    """

    global_weights = global_model.state_dict()

    # Calculate scores for each set of local weights based on cosine similarity
    scores = [cosine_similarity_score(local_weight, global_weights) for local_weight in local_weights]

    # Sort the local weights by their scores
    sorted_weights_scores = sorted(zip(local_weights, scores), key=lambda x: x[1], reverse=True)

    # Trim the worst-performing weights based on the trim_ratio
    trim_count = int(len(sorted_weights_scores) * trim_ratio)
    trimmed_weights = [w for w, _ in sorted_weights_scores[trim_count:-trim_count]]

    # Select the best aggregation method based on the highest score
    best_aggregation_method_name = max(aggregation_methods, key=lambda k: sorted_weights_scores[0][1])
    best_aggregation_method = aggregation_methods[best_aggregation_method_name]

    # Aggregate the trimmed list of weights using the best method
    aggregated_weights = best_aggregation_method(trimmed_weights)

    return aggregated_weights