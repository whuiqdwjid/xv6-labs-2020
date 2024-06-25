import torch
import numpy as np
import math
import copy
from collections import OrderedDict

def disturbance1(local_weights, choice, distur_ratio):
    """
    对一部分本地权重应用三种扰动类型中的一种

    Args:
    - local_weights (list of OrderedDict): List of benign gradients (∇b) as OrderedDicts.
    - choice (str): The type of disturbance to apply ('uv', 'std', 'sgn').
    - distur_ratio (float): The ratio of weights to disturb.

    Returns:
    - list of OrderedDict: List of disturbed local weights.
    """
    n = len(local_weights)
    num_disturb = int(n * distur_ratio)
    disturb_indices = np.random.choice(n, num_disturb, replace=False)

    # 对于sgn扰动，首先计算所有权重的全局平均值
    if choice == 'sgn':
        all_weights = torch.cat([weight for sublist in local_weights for weight in sublist.values()])
        global_avg = torch.mean(all_weights)
        global_sign = torch.sign(global_avg)

    disturbed_weights = []
    r = 0.1  # 扰动比例因子
    for i, weight_dict in enumerate(local_weights):
        disturbed_weight_dict = OrderedDict()
        for key, weight in weight_dict.items():
            if i in disturb_indices:
                if choice == 'uv':
                    disturbed_weight_dict[key] = weight - r * weight / torch.norm(weight)
                elif choice == 'std':
                    std_dev = torch.std(weight)
                    disturbed_weight_dict[key] = weight + r * (-std_dev)
                elif choice == 'sgn':
                    # 应用全局平均值的符号扰动
                    disturbed_weight_dict[key] = weight + r * (-global_sign)
            else:
                disturbed_weight_dict[key] = weight
        disturbed_weights.append(disturbed_weight_dict)

    return disturbed_weights

# 示例使用
# disturbed_weights = disturbance1(local_weights, 'sgn', 0.1)


# disturbed_weights = disturbance(local_weights, 'uv', 0.1)
def disturbance_global(global_weights, perturbation, r):
        disturbed_weights = {}
        if perturbation == 'uv':
            norm = torch.norm(torch.cat([w.view(-1) for w in global_weights.values()]))
            for key, weight in global_weights.items():
                disturbed_weights[key] = weight - r * weight / norm
        elif perturbation == 'std':
            for key, weight in global_weights.items():
                std_dev = torch.std(weight)
                disturbed_weights[key] = weight + r * (-std_dev)
        elif perturbation == 'sgn':
            global_mean = torch.mean(torch.cat([w.view(-1) for w in global_weights.values()]))
            global_sign = torch.sign(global_mean)
            for key, weight in global_weights.items():
                disturbed_weights[key] = weight + r * (-global_sign)

        return disturbed_weights

def cal_max_distance(local_weights):
    norms = []
    max_distance = {}
    # 对每个key
    for key in local_weights[0].keys():
        # 计算指定key下，每个local的之间的2范数
        # 计算每对local值之间的 2 范数 找出所有 2 范数的最大值
        norms = [torch.norm(local_weights[i][key], p=2) for i in range(0, len(local_weights))]
        max_distance[key] = max(norms)#每个key选最大的
    return max_distance

def cal_max_sum_distance(local_weights):
    norms = []
    sum_distance = []
    max_sum_distance = {}
    # 对每个key
    for key in local_weights.keys():
        for i in range(0, len(local_weights)):
            norms = [torch.norm(local_weights[j][key], p=2) for j in range(i, len(local_weights))]
            #加上2模平方
            sum_distance.append(sum(norms)-norms[i])
        max_sum_distance[key] = max(sum_distance)
    return sum_distance

def judge_distance(Malicious_nodes_weight,Benign_node_weights,max_distance):
    d_n_distance = {}
    for i in range(0, len(Benign_node_weights)):
        # for key, weight in local_weights[i].items():
        dis = torch.norm((Malicious_nodes_weight['layer_hidden.bias'] - Benign_node_weights[i]['layer_hidden.bias']), p=2)
        # print(dis)
        d_n_distance['layer_hidden.bias'] = dis
        if (d_n_distance['layer_hidden.bias'] > max_distance['layer_hidden.bias']):
            return True
    return False

def judge_sum_distance(local_weights,disturbed_weights,sum_distance):
    d_n_distance = {}
    for i in range(0, len(local_weights)):
        for key, weight in local_weights[i]:
            if(i==0):
                d_n_distance[key] = 0
            d_n_distance[key] += torch.norm((local_weights[i][key] - disturbed_weights[key]), p=2)
            if (d_n_distance[key] > sum_distance[key]):
                return True
    return False

def min_max_attack(Malicious_nodes_weight, Benign_node_weights, perturbation, r):
    #计算良性向量最大距离
    maxtimes = 20
    max_distance = cal_max_distance(Benign_node_weights)
    Rsucc = 0
    tao = 0.01
    step = r/2
    #初始值扰动系数下的恶意向量
    changed_Malicious_weights = disturbance_global(Malicious_nodes_weight, perturbation, r)

    while(abs(Rsucc-r)>tao and maxtimes>0):
        # print('r:{},succ:{}'.format(r, Rsucc))
        #判断恶意向量与任意良性向量的距离是否小于最大的良性向量距离
        isExceed = judge_distance(changed_Malicious_weights,Benign_node_weights,max_distance)
        if(isExceed):
            Rsucc = r
            r = r + step / 2
        else:
            r = r - step / 2
        #得到新的r，重新计算一次扰动
        changed_Malicious_weights = disturbance_global(Malicious_nodes_weight, perturbation, r)
        step = step / 2
        maxtimes = maxtimes - 1

    return Rsucc,changed_Malicious_weights

def min_sum_attack(global_weights, local_weights, perturbation, r):
    #先用 预设的r设置一个扰动后的结果得到disturbed_weights
    disturbed_weights = disturbance_global(global_weights, perturbation, r)
    #计算最大距离
    max_sum_distance = cal_max_sum_distance(local_weights)
    #判断是否有超出情况
    #isExceed = judge_distance(local_weights,disturbed_weights,max_distance)
    Rsucc = 0
    tao = 0.01
    step = r/2
    while(abs(Rsucc-r)>tao):
        isExceed = judge_sum_distance(local_weights,disturbed_weights,max_sum_distance)
        if(isExceed):
            Rsucc = r
            r = r + step / 2
        else:
            r = r - step / 2
        #得到新的r，重新计算一次扰动
        disturbed_weights = disturbance_global(global_weights, perturbation, r)
        step = step / 2
    return Rsucc