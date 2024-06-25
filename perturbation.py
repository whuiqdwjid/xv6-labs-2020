# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import argparse
import torch
from tensorboardX import SummaryWriter


from k_means_constrained import KMeansConstrained
from collections import defaultdict
from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from collections import defaultdict
from sklearn.linear_model import LinearRegression

#聚合算法：krum、bulyan、median_aggregate、trimmed_mean_aggregate
from utils import get_dataset, exp_details,CreateLinearRegression
from aggregation import average_weights,krum,median_aggregate,bulyan,trimmed_mean_aggregate,fang_aggregated,projection_weights
from perturbation import disturbance1,disturbance_global,min_sum_attack,min_max_attack
from scipy.optimize import curve_fit

import matplotlib
matplotlib.use('Agg')  # 远程只能用Agg，安装qt没试
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def logarithmic_func(x,a,b):
    # print(a,b,c)
    # print(x)
    return a*np.log(b * x+ 1e-10)

def exchange_group_by_rank(epoch,local_weights,history_group,history_rank):
    Local_aggregation_weight = []
    new_group = []
    if epoch<3:
        for key,value in local_weights.items():  # 每个usr都update_weights，并把每批次训练的 w, loss存储
            # 每个用户的本地模型
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[key], logger=logger)

            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model_copy), global_round=epoch)
            local_weights[idxs_users[idx_index]] = copy.deepcopy(w)
            local_losses.append(copy.deepcopy(loss))

    return Local_aggregation_weight,new_group

if __name__ == '__main__':
    start_time = time.time()
    print(matplotlib.get_backend())
    print(matplotlib.get_backend())


    # define paths 设置日志保存路径
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    #解析参数，参数在option中设置
    args = args_parser()
    exp_details(args)
    #分辨参数中使用cpu还是gpu
    if args.gpu:
        torch.cuda.set_device(args.gpu)
        print("this is GPU")
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups 加载数据集用utils中的get_dataset函数选取训练集和测试集，并根据用户数量将训练集划分
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape#获取图片尺寸784
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,#输入尺寸为图片尺寸长乘宽，通过一个映射层到64，再映射到0-9上
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)#global_model是要被训练的模型，传递到device
    global_model.train()#开启模型训练模式
    # MLP(
    #     (layer_input): Linear(in_features=784, out_features=64, bias=True)
    # (relu): ReLU()
    # (dropout): Dropout(p=0.5, inplace=False)
    # (layer_hidden): Linear(in_features=64, out_features=10, bias=True)
    # (softmax): Softmax(dim=1)
    # )

    # copy weights
    global_weights = global_model.state_dict()

    shapes_dict = {}
    for key, value in global_weights.items():
        shapes_dict[key] = value.shape if hasattr(value, 'shape') else "Not a Numpy array"
    print(shapes_dict)
    #全局权重
    # OrderedDict([('layer_input.weight', tensor([[-0.0052, 0.0338, -0.0346, ..., 0.0188, -0.0147, -0.0317],
    #                                             [-0.0268, -0.0338, -0.0086, ..., 0.0126, 0.0084, -0.0287],
    #                                             [-0.0332, 0.0345, 0.0151, ..., 0.0066, -0.0204, -0.0213],
    #                                             ...,
    #                                             [0.0349, -0.0273, -0.0115, ..., 0.0271, 0.0344, -0.0326],
    #                                             [-0.0109, -0.0279, 0.0184, ..., 0.0256, 0.0091, 0.0102],
    #                                             [-0.0348, 0.0058, -0.0305, ..., -0.0210, -0.0141, 0.0170]])),
    #              ('layer_input.bias', tensor([0.0117, -0.0194, -0.0335, -0.0122, 0.0220, -0.0231, 0.0290, 0.0115,
    #                                           0.0189, -0.0324, 0.0226, -0.0339, -0.0084, 0.0080, 0.0110, 0.0085,
    #                                           0.0053, -0.0239, 0.0313, 0.0278, -0.0169, 0.0249, 0.0103, 0.0127,
    #                                           0.0137, -0.0249, 0.0184, 0.0144, -0.0160, 0.0348, 0.0080, 0.0312,
    #                                           0.0128, -0.0280, 0.0298, 0.0326, 0.0082, 0.0325, 0.0178, 0.0105,
    #                                           -0.0139, -0.0197, -0.0146, -0.0116, -0.0091, 0.0014, -0.0080, -0.0353,
    #                                           -0.0040, -0.0102, 0.0145, 0.0049, 0.0166, 0.0224, 0.0206, 0.0037,
    #                                           0.0152, -0.0328, -0.0014, -0.0336, 0.0079, -0.0104, 0.0090, -0.0171])),
    #              ('layer_hidden.weight', tensor([[9.7278e-02, -3.9755e-03, -3.2692e-02, 1.3977e-02, -3.3897e-02,

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    group_max_acc_list = []

    #画图使用
    every_epoch_acc_plt = []


    #分组数据
    #id_in_group = [[0 for _ in range(args.group_size)] for _ in range(args.group_num)]
    #group_performance = []

    # 从用户中选出m个，计算用于FL的用户数量，用户的id暂时为顺序增加的数字
    m = max(int(args.frac * args.num_users), 1)

    # 选出id后不再改变,暂时设置业务场景为选中就始终为这些节点
    idxs_users = np.random.choice(range(args.num_users), m, replace=False)

    global_model_copy=copy.deepcopy(global_model)
    ord_infere_test_acc=[]
    ord_train_accuracy=[]

    # 选出拜占庭节点
    byz_num= int(m * args.byz_ratio)
    byz_users = np.random.choice(idxs_users,byz_num , replace=False)

    print('ttttttttttttttttttttttttttttest')
    #初始原版联邦学习，同样输入下的准确率图像
    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = {}, []
        print(f'\n | Global Training Round : {epoch + 1} |\n')

        global_model_copy.train()
        # 本地模型训练，并区分良性和恶意模型
        Malicious_nodes_weights = {}
        Benign_node_weights = []
        print('idxs_users:{}'.format(idxs_users))
        print('byz_users:{}'.format(byz_users))
        for idx_index in range(0, len(idxs_users)):  # 每个usr都update_weights，并把每批次训练的 w, loss存储
            # 每个用户的本地模型
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idxs_users[idx_index]], logger=logger)
            # print(user_groups[idx])
            # print(type(train_dataset))
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model_copy), global_round=epoch)
            local_weights[idxs_users[idx_index]] = copy.deepcopy(w)
            local_losses.append(copy.deepcopy(loss))
            if idxs_users[idx_index] in byz_users:
                print('curent id:{},is Malicious node'.format(idxs_users[idx_index]))
                Malicious_nodes_weights[idxs_users[idx_index]] = copy.deepcopy(w)
            else:
                # print('curent id:{},is Benign node'.format(idxs_users[idx_index]))
                Benign_node_weights.append(copy.deepcopy(w))

        # 添加扰动向量
        # for key, value in Malicious_nodes_weights.items():  # 每个usr都update_weights，并把每批次训练的 w, loss存储
        #     # 未知服务器聚合规则和良性向量下，使用min_max或min_sum计算真正的扰动系数r
        #     # print('curent id:{},Malicious node'.format(key))
        #     real_r, changed_Malicious_weights = min_max_attack(value, Benign_node_weights, args.perturbation, args.r)
        #     print('this is real r:{}'.format(real_r))
        #     # 将本地模型替换成恶意模型
        #     local_weights[key] = changed_Malicious_weights

        #重新计算带有扰动向量的全局模型
        disturbed_weights = []
        for key,value in local_weights.items():
            disturbed_weights.append(value)
            # print(type(value))
        global_weights_copy = average_weights(disturbed_weights)  # global_weights在外边定义，每一批次训练都对其进行更新
        # print(global_weights)
        print("\n\n\n\n")
        # update global weights
        global_model_copy.load_state_dict(global_weights_copy)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model_copy.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[0], logger=logger)
            acc, loss = local_model.inference(model=global_model_copy)
            list_acc.append(acc)
            list_loss.append(loss)
        ord_train_accuracy.append(sum(list_acc) / len(list_acc))
        ord_test_acc, ord_test_loss = test_inference(args, global_model_copy, test_dataset)
        ord_infere_test_acc.append(ord_test_acc)
        # print global training loss after every 'i' rounds
        if (epoch + 1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100 * ord_train_accuracy[-1]))
    # Test inference after completion of training
    ord_test_acc, ord_test_loss = test_inference(args, global_model_copy, test_dataset)
    train_loss.clear()
    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100 * ord_train_accuracy[-1]))  # 输出列表的最后一位
    print("|---- Test Accuracy: {:.2f}%".format(100 * ord_test_acc))

    # Saving the objects train_loss and train_accuracy:
    file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'. \
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n ordinary FL Total Run Time: {0:0.4f}\n\n\n'.format(time.time() - start_time))


    for epoch in tqdm(range(args.epochs)):#epoch从0开始
        # local_weights, local_losses = [], []
        local_losses = []
        local_weights = {}
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()

        #本地模型训练，并区分良性和恶意模型
        Malicious_nodes_weights = {}
        Benign_node_weights = []
        for idx_index in range(0, len(idxs_users)):  # 每个usr都update_weights，并把每批次训练的 w, loss存储
            #每个用户的本地模型
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idxs_users[idx_index]], logger=logger)
            # print(user_groups[idx])
            # print(type(train_dataset))
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            local_weights[idxs_users[idx_index]] = copy.deepcopy(w)
            local_losses.append(copy.deepcopy(loss))
            if idxs_users[idx_index] in byz_users:
                Malicious_nodes_weights[idxs_users[idx_index]]=copy.deepcopy(w)
            else:
                Benign_node_weights.append(copy.deepcopy(w))

        #添加扰动向量
        for key,value in Malicious_nodes_weights.items():  # 每个usr都update_weights，并把每批次训练的 w, loss存储
            # 未知服务器聚合规则和良性向量下，使用min_max或min_sum计算真正的扰动系数r
            real_r,changed_Malicious_weights = min_max_attack(value, Benign_node_weights, args.perturbation, args.r)
            print('this is real r:{}'.format(real_r))
            # 将本地模型替换成恶意模型
            local_weights[key] = changed_Malicious_weights

        ###################################################
        ############            分层分组         ############
        ###################################################
        #idxs_users是id存储位置
        result_dict = {}
        #PCA会导致输出结果大小不一致，不加
        transformed_data = []
        for i in range(len(idxs_users)):
            result_dict[idxs_users[i]] = local_weights[idxs_users[i]]['layer_hidden.weight']

        shapes = [arr.shape for arr in result_dict.values()]
        clustered_values = []
        if len(set(shapes)) == 1:
            # 提取值并组合成一个数组
            data = np.array([np.array(tensor) for tensor in result_dict.values()])
            # 将二维数组展平为一维
            flattened_data = data.reshape(data.shape[0], -1)

            # 定义要聚类的类别数
            num_clusters = 5
            # 使用平衡 K-Means 聚类
            # size_min 和 size_max 定义了每个聚类的最小和最大大小
            size_min = len(flattened_data) // num_clusters  # 确保每个聚类至少有这么多元素
            size_max = (len(flattened_data) + num_clusters - 1) // num_clusters  # 确保不会有聚类超过这么多元素

            kmeans = KMeansConstrained(
                n_clusters=num_clusters,
                size_min=size_min,
                size_max=size_max,
                random_state=42
            )
            kmeans.fit(flattened_data)
            cluster_ids = kmeans.labels_

            value_index_dict = defaultdict(list)
            # 将索引按照值进行分组
            for index, value in enumerate(cluster_ids):
                value_index_dict[value].append(list(result_dict.keys())[index])

            # 输出每组的索引
            for value, indices in value_index_dict.items():
                print("Value", value, ":", indices)

            for i in range(len(value_index_dict)):
                row = []
                # print(value_index_dict[i])
                for j in range(len(value_index_dict[i])):
                    # print(value_index_dict[i][j])+
                    row.append(local_weights[value_index_dict[i][j]])
                clustered_values.append(row)
        else:
            print("Arrays have different shapes and cannot be clustered together.")

        #保存组内聚合结果
        group_aggregation_results = []
        group_length=[]
        tmp_weights=[]


        ###################################################
        ############            组内聚合         ############
        ###################################################

        for key_value in clustered_values:
            if(len(key_value))==0:
                continue
            if (args.aggregation == "average_weights"):
                tmp_weights = average_weights(key_value)
            if (args.aggregation == "krum"):
                tmp_weights = krum(clustered_values)
            if (args.aggregation == "trimmed_mean_aggregate"):
                trim_ratio = 0.1  # 去除每侧 10% 的数据
                tmp_weights = trimmed_mean_aggregate(key_value, trim_ratio)
            if (args.aggregation == "bulyan"):
                bulyan(key_value, int(0.3 * len(key_value)))
            if args.aggregation == "fang_aggregated":
                tmp_weights = fang_aggregated(
                    local_weights=key_value,
                    global_model=global_model,
                    aggregation_methods={
                        'krum': krum,
                        'average_weights': average_weights,
                        'trimmed_mean_aggregate': trimmed_mean_aggregate,
                        "bulyan": bulyan
                    },
                    device=device,
                    trim_ratio=0.1
                )
            group_length.append(len(key_value)/m)
            group_aggregation_results.append(tmp_weights)
        print(group_length)
        # print(len(group_aggregation_results))
        #计算组内聚合结果的准确率并保存
        group_acc=[]


        if (epoch < 3):
            for each_weight in group_aggregation_results:
                print(f'epoch {epoch} calculate group weight')
                local_model_tmp = global_model
                local_model_tmp.load_state_dict(each_weight)
                local_model = LocalUpdate(args=args, dataset=test_dataset,
                                          idxs=[], logger=logger)  # 这里传group的索引
                test_acc, test_loss = local_model.test_dataset_inference(model=local_model_tmp)
                group_acc.append(test_acc)

            # 找到最大的准确率值
            Benchmark = max(group_acc)
            Benchmark_index = group_acc.index(Benchmark)
            group_max_acc_list.append(Benchmark)
            every_epoch_acc_plt.append(Benchmark)

            # 权重计算
            theta_t = []
            for i in range(0, len(group_aggregation_results)):
                # print(group_aggregation_results[i])

                if i != Benchmark_index:
                    # 计算欧氏距离
                    distance = torch.norm(
                        torch.tensor(group_aggregation_results[i]['layer_hidden.weight']) - torch.tensor(
                            group_aggregation_results[Benchmark_index]['layer_hidden.weight']))
                    # 计算余弦相似度
                    dot_product = np.dot(group_aggregation_results[i]['layer_hidden.weight'].flatten(),
                                         group_aggregation_results[Benchmark_index]['layer_hidden.weight'].flatten())
                    norm_array1 = np.linalg.norm(group_aggregation_results[i]['layer_hidden.weight'])
                    norm_array2 = np.linalg.norm(group_aggregation_results[Benchmark_index]['layer_hidden.weight'])
                    cosine_similarity = dot_product / (norm_array1 * norm_array2)
                    theta_t.append(cosine_similarity * distance)
                else:
                    theta_t.append(1)
            # sigmod投影
            sigmoid_values = [sigmoid(theta_t[i]) * (group_length[i] / 20) for i in range(0, len(theta_t))]
            # 归一化处理，使权重加和为1
            projection = sigmoid_values / np.sum(sigmoid_values)
            print(f'epoch {epoch} processing projection aggregation')
            global_weights = projection_weights(group_aggregation_results, projection)
            # print(group_acc)


        else:
            # 拟合预测准确率曲线
            X = [i for i in range(0, len(group_max_acc_list) - 1)]
            Y = [group_max_acc_list[i] - group_max_acc_list[i - 1] for i in range(1, len(group_max_acc_list))]
            print(X)
            print(Y)
            popt, pcov = curve_fit(f=logarithmic_func, xdata=X, ydata=Y, p0=[-1.0, 1.0])  # popt返回值是残差最小时的参数，即最佳参数 ,2.0
            y_pred = logarithmic_func(len(group_max_acc_list) + 1, popt[0], popt[1])  # 将x和参数带进去，得到y的预测值 ,popt[2]
            if y_pred < 0:
                every_epoch_acc_plt.append(group_max_acc_list[-1] + group_max_acc_list[-1] - group_max_acc_list[-2])
            else:
                every_epoch_acc_plt.append(group_max_acc_list[-1] + y_pred)
            divider = y_pred

            for each_weight in group_aggregation_results:
                print(f'epoch {epoch} calculate group weight')
                local_model_tmp = global_model
                local_model_tmp.load_state_dict(each_weight)
                local_model = LocalUpdate(args=args, dataset=test_dataset,
                                          idxs=[], logger=logger)  # 这里传group的索引
                test_acc, test_loss = local_model.test_dataset_inference(model=local_model_tmp)
                group_acc.append(test_acc)
            # 找到最大的准确率值
            Benchmark = max(group_acc)
            Benchmark_index = group_acc.index(Benchmark)
            group_max_acc_list.append(Benchmark)

            # 权重计算
            theta_t = []
            for i in range(0, len(group_aggregation_results)):
                # print(group_aggregation_results[i])

                if i != Benchmark_index:
                    # 计算欧氏距离
                    distance = torch.norm(
                        torch.tensor(group_aggregation_results[i]['layer_hidden.weight']) - torch.tensor(
                            group_aggregation_results[Benchmark_index]['layer_hidden.weight']))
                    # 计算余弦相似度
                    dot_product = np.dot(group_aggregation_results[i]['layer_hidden.weight'].flatten(),
                                         group_aggregation_results[Benchmark_index]['layer_hidden.weight'].flatten())
                    norm_array1 = np.linalg.norm(group_aggregation_results[i]['layer_hidden.weight'])
                    norm_array2 = np.linalg.norm(group_aggregation_results[Benchmark_index]['layer_hidden.weight'])
                    cosine_similarity = dot_product / (norm_array1 * norm_array2)
                    theta_t.append(cosine_similarity * distance)
                else:
                    theta_t.append(1)
            # sigmod投影
            sigmoid_values = [sigmoid(theta_t[i]) * (group_length[i] / 20) for i in range(0, len(theta_t))]
            # 归一化处理，使权重加和为1
            projection = sigmoid_values / np.sum(sigmoid_values)
            print(f'epoch {epoch} processing projection aggregation')
            global_weights = projection_weights(group_aggregation_results, projection)
            # print(group_acc)
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()

        # 直接用测试集进行测试，不对每个目标的数据集中的测试集测试
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        train_accuracy.append(test_acc)

        # for c in range(args.num_users):
        #     local_model = LocalUpdate(args=args, dataset=test_dataset,
        #                               idxs=user_groups[idx], logger=logger)
        #     acc, loss = local_model.test_dataset_inference(model=global_model)#LocalUpdate中输入测试集，inference函数中计算输入model的loss 和准确率
        #     list_acc.append(acc)
        #     list_loss.append(loss)
        # train_accuracy.append(sum(list_acc)/len(list_acc))
        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))


    #这里跳出了循环
    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # Saving the objects train_loss and train_accuracy:
    file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    print(every_epoch_acc_plt)
    print(train_accuracy)
    epoch_list = list(range(1,len(every_epoch_acc_plt)+1))
    print(epoch_list)
    # epoch_list=[1,2,3,4,5,6,7,8,9,10]
    plt.plot(epoch_list,every_epoch_acc_plt,label='Benchmark value (prediction accuracy)')
    plt.plot(epoch_list, train_accuracy, label='true accuracy）')
    plt.plot(epoch_list,ord_infere_test_acc,label='ordinary FL accuracy')
    plt.legend()
    plt.title('three accrucy compare')
    plt.xlabel('epoch')
    plt.ylabel('accracy')
    plt.savefig('./num{}_byz{}_agr-updates_{}.png'.format(m,byz_num,args.perturbation))
    print('num{}_byz{}_agr-updates_{}.png'.format(m,byz_num,args.perturbation))
    print('ok')
    plt.close()
    # plt.show()
    # #PLOTTING (optional)
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('Agg')
    #
    # # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('D:/任务/研一项目/论文/Federated - Learning - PyTorch - master/save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    #
    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))

# input:group_num,group1[0..p]~groupn[0..p]
# for i<-n to 2 do
#     if max(groupi)>max(groupi-1)
#         swap max(groupi) and min(groupi-1)
# for i<1 to n-1 do
#     if min(groupi)<min(groupi+1)
#         swap min(groupi) and max(groupi+1)
# output:group1'[0..p]~groupn'[0..p]

# ###################################################
# ############            组间聚合         ############
# ###################################################
# if(epoch<5):
#     for each_weight in group_aggregation_results:
#         print(f'epoch {epoch} calculate group weight')
#         local_model_tmp=global_model
#         local_model_tmp.load_state_dict(each_weight)
#         local_model = LocalUpdate(args=args, dataset=test_dataset,
#                                   idxs=[], logger=logger)  # 这里传group的索引
#         test_acc, test_loss = local_model.test_dataset_inference(model=local_model_tmp)
#         group_acc.append(test_acc)
#
#     # 找到最大的准确率值
#     Benchmark = max(group_acc)
#     Benchmark_index = group_acc.index(Benchmark)
#     group_max_acc_list.append(Benchmark)
#     every_epoch_acc_plt.append(Benchmark)
#
#     #权重计算
#     theta_t = []
#     for i in range(0,len(group_aggregation_results)):
#         print(group_aggregation_results[i])
#
#         if i!=Benchmark_index:
#             # 计算欧氏距离
#             distance = torch.norm(torch.tensor(group_aggregation_results[i]['layer_hidden.weight']) - torch.tensor(
#                 group_aggregation_results[Benchmark_index]['layer_hidden.weight']))
#             #计算余弦相似度
#             dot_product = np.dot(group_aggregation_results[i]['layer_hidden.weight'].flatten(), group_aggregation_results[Benchmark_index]['layer_hidden.weight'].flatten())
#             norm_array1 = np.linalg.norm(group_aggregation_results[i]['layer_hidden.weight'])
#             norm_array2 = np.linalg.norm(group_aggregation_results[Benchmark_index]['layer_hidden.weight'])
#             cosine_similarity = dot_product / (norm_array1 * norm_array2)
#             theta_t.append(cosine_similarity*distance)
#         else:
#             theta_t.append(1)
#     #sigmod投影
#     sigmoid_values = [sigmoid(theta_t[i])*(group_length[i]/20) for i in range(0,len(theta_t))]
#     # 归一化处理，使权重加和为1
#     projection = sigmoid_values / np.sum(sigmoid_values)
#     #clustered_values需要变成一个列表记得改！
#     print(f'epoch {epoch} processing projection aggregation')
#     global_weights =projection_weights(group_aggregation_results,projection)
#     # print(group_acc)
#
#
# else:
#     # 制作拟合函数的输入
#     # print(len(group_max_acc_list))
#     X = [i for i in range(0,len(group_max_acc_list)-1)]
#     Y = [group_max_acc_list[i] - group_max_acc_list[i - 1] for i in range(1, len(group_max_acc_list))]
#     print(X)
#     print(Y)
#     popt, pcov = curve_fit(f=logarithmic_func, xdata=X, ydata=Y, p0=[-1.0, 1.0])  # popt返回值是残差最小时的参数，即最佳参数 ,2.0
#     y_pred = logarithmic_func(len(group_max_acc_list)+1, popt[0], popt[1])  # 将x和参数带进去，得到y的预测值 ,popt[2]
#     if y_pred<group_max_acc_list[-1] - group_max_acc_list[-2]:
#         every_epoch_acc_plt.append(group_max_acc_list[-1] + group_max_acc_list[-1] - group_max_acc_list[-2])
#     else:
#         every_epoch_acc_plt.append(group_max_acc_list[-1]+y_pred)
#     divider = y_pred
#
#     for each_weight in group_aggregation_results:
#         print(f'epoch {epoch} calculate group weight')
#         local_model_tmp = global_model
#         local_model_tmp.load_state_dict(each_weight)
#         local_model = LocalUpdate(args=args, dataset=test_dataset,
#                                   idxs=[], logger=logger)  # 这里传group的索引
#         test_acc, test_loss = local_model.test_dataset_inference(model=local_model_tmp)
#         group_acc.append(test_acc)
#
#     ratio_list = [x / divider for x in group_acc]
#     modified_list = [1.1 if x > 1.1 else (0.9 if x < 0.9 else x) for x in ratio_list]
#     # sigmod投影
#     # sigmoid_values = [sigmoid(val) for val in group_acc]
#     sigmoid_values = [sigmoid(group_acc[i]) * group_length[i] for i in range(0, len(group_acc))]
#     # 乘比率
#     sigmoid_values = [ratio * val for ratio,val in zip(ratio_list,sigmoid_values)]
#     # 归一化处理，使权重加和为1
#     projection = sigmoid_values / np.sum(sigmoid_values)
#     # clustered_values需要变成一个列表记得改！
#     print(f'epoch {epoch} processing projection aggregation')
#     global_weights = projection_weights(group_aggregation_results, projection)
#
#     #找到最大的准确率值
#     if len(group_max_acc_list)==5:
#         group_max_acc_list.pop(0)
#         Benchmark = max(group_acc)
#         group_max_acc_list.append(Benchmark)
###################################################
############            组间聚合         ############
###################################################


