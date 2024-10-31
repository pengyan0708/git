import torch
import numpy as np
import heapq
import default

'''
对于每个样本，m个clients都会生成对应的的logits，其中有c个恶意clients，
用欧几里得距离为每个client找到m-c-2个与其最近的logits，
最后选择与所有其他clients欧几里得距离之和最小的那个client的logits作为全局的agg_logits；
（对logits整体）
'''


import torch

def Krum(temp_all_result, args):
    num_clients = len(temp_all_result)
    num_malicious = args.nbyz

    clients_l2 = [[] for _ in range(num_clients)]

    # 计算每个客户端与其他客户端的欧几里得距离
    for i in range(num_clients):
        for j in range(num_clients):
            if i != j:
                l2_distance = torch.dist(temp_all_result[i], temp_all_result[j], p=2)
                clients_l2[i].append((l2_distance, j))

    # 选择距离最小的 m-c-2 个客户端
    k = num_clients - num_malicious - 2
    selected_indices = []
    for i in range(num_clients):
        sorted_distances = sorted(clients_l2[i], key=lambda x: x[0])
        closest_indices = [idx for dist, idx in sorted_distances[:k]]
        selected_indices.append(closest_indices)

    # 计算被选客户端的投票结果
    votes = []
    for indices in selected_indices:
        vote = torch.zeros_like(temp_all_result[0])
        for idx in indices:
            vote += temp_all_result[idx]
        votes.append(vote)

    # 选择投票结果中 L2 距离最小的客户端作为全局聚合结果
    min_distance = float('inf')
    global_aggregate = None
    for vote in votes:
        l2_distance = torch.dist(vote, torch.zeros_like(vote), p=2)
        if l2_distance < min_distance:
            min_distance = l2_distance
            global_aggregate = vote

    return global_aggregate



'''
对于每个样本，m个clients都会生成对应的的logits，其中有c个恶意clients，独立地生成c个恶意的logits。
对于每个样本的logits，剔除m个clients中最大和最小的β个logits（β≥c），
然后计算剩下的m-2β个值的logits均值；
（对样本logits的每一维）
'''


def transpose(matrix):
    new_matrix = []
    for i in range(len(matrix[0])):
        matrix1 = []
        for j in range(len(matrix)):
            matrix1.append(matrix[j][i])
        new_matrix.append(matrix1)
    return new_matrix


def Trimmed_mean(temp_all_result, args):
    # beta = args.user_number  - args.benign_user_number      #beta目前等于恶意用户的数目
    # # 对于每个样本，m个clients都会生成对应的的logits;一共有batch个labels，用labels_logits表示
    # labels_logits = [[] for _ in range(len(temp_all_result[1]))]
    #
    #
    # # 为每个label都找到最佳的logits
    # for index1, label_logits1 in enumerate(temp_all_result[0]):
    #     labels_temp = []
    #     for index2, client_logits in enumerate(temp_all_result):
    #         labels_temp.append(temp_all_result[index2][index1])
    #     labels_temp = labels_temp.transpose()           #矩阵转置一下，直接就是 user_number X logits_number ==> logits_number X user_number
    #     labels_temp = list.sort(labels_temp)
    #     labels_logits[index1] = labels_temp[:,beta:-beta].mean(axis=1)       #截去掉最小的beta和最大的beta个数,然后求他们的平均;存到labels_logits中
    #
    # agg_avg_labels = labels_logits

    beta = args.nworkers - args.nbyz  # beta目前等于恶意用户的数目
    labels_logits = [[] for _ in range(len(temp_all_result[0]))]
    for label_index1, label_logits1 in enumerate(temp_all_result[0]):  # enumerate(temp_all_result[0])表示枚举每一个label样本
        labels_temp = []
        for demission_index2, label_dimensions in enumerate(
                temp_all_result[0][0]):  # enumerate(temp_all_result[0][0])相当于枚举每一个维度
            labels_dimission_temp = []  # 记录单个label的每一维的平均
            for user_index3, client_logits in enumerate(
                    temp_all_result):  # enumerate(temp_all_result)表示枚举每一个clients用户——相当于把每一列都取出来
                # print(len(temp_all_result[index2][index1]))
                labels_dimission_temp.append(temp_all_result[user_index3][label_index1][
                                                 demission_index2])  # temp_all_result[index2][index1][index3]索引到每个用户的每个样本的每个维度
            list.sort(labels_dimission_temp)
            # print("labels_dimission_temp:", type(labels_dimission_temp), labels_dimission_temp[beta:-beta])
            labels_temp.append(sum(labels_dimission_temp[beta:-beta]) / len(
                labels_dimission_temp[beta:-beta]))  # 所有client针对某个label的某一维数据求平均
        labels_logits[label_index1] = labels_temp
    agg_avg_labels = torch.Tensor(labels_logits)  # 把list转成tensor
    return agg_avg_labels
    # import copy
    # # Trimmed_mean
    # beta = args.user_number - args.benign_user_number  # beta目前等于恶意用户的数目
    # # 对于每个样本，m个clients都会生成对应的的logits;一共有batch个labels，用labels_logits表示
    # labels_logits = [[] for _ in range(len(temp_all_result[0]))]
    # # print(len(labels_logits))
    #
    # # 为每个label都找到最佳的logits
    # for index1, label_logits1 in enumerate(temp_all_result[0]):  # enumerate(temp_all_result[0])表示枚举每一个label样本
    #     labels_temp = []
    #
    #     for index3, label_dimensions in enumerate(temp_all_result[0][0]):  # enumerate(temp_all_result[0][0])相当于枚举每一个维度
    #         labels_dimission_temp = []  # 记录单个label的每一维的平均
    #         for index2, client_logits in enumerate(
    #                 temp_all_result):  # enumerate(temp_all_result)表示枚举每一个clients用户——相当于把每一列都取出来
    #             # print(len(temp_all_result[index2][index1]))
    #             labels_dimission_temp.append(
    #                 temp_all_result[index2][index1][index3])  # temp_all_result[index2][index1][index3]索引到每个用户的每个样本的每个维度
    #         print(labels_dimission_temp[beta:-beta])
    #         labels_temp.append(labels_dimission_temp[beta:-beta])  # 所有client针对某个label的某一维数据求平均
    #     # print(len(labels_temp))
    #     # labels_temp = labels_temp.transpose()           #矩阵转置一下，直接就是 user_number X logits_number ==> logits_number X user_number
    #     # print(labels_temp)
    #     # for index, label_temp in enumerate(labels_temp):    #矩阵转置一下，直接就是 user_number X logits_number ==> logits_number X user_number
    #     #     # print("label_temp:",label_temp)
    #     #     label_temp2 = torch.sort(label_temp.clone().detach())
    #     #     labels_temp2.append(label_temp2)
    #     #     # print("label_temp:",label_temp)
    #     # # print(labels_temp2)
    #     # print(labels_temp2[0][0])
    #     # labels_temp2 = list(map(list, zip(*labels_temp)))
    #     # # labels_temp2 = copy.deepcopy(list(map(list, zip(*labels_temp))) )              #矩阵转置一下，直接就是 user_number X logits_number ==> logits_number X user_number
    #     # print(labels_temp2,'\n')
    #     # list.sort(labels_temp2)                  #把list排序一下，为了后面的切片 logits X user
    #     # print(labels_temp)
    #     # print(labels_temp2)
    #     labels_logits[index1] = labels_dimission_temp
    #     # labels_logits[index1] = labels_temp2[:,beta:-beta].mean(axis=1)       #截去掉最小的beta和最大的beta个数logits,然后求他们的平均;存到labels_logits中


'''
结合了Krum和Trimmed mean的一个变体。先用Krum找出 x个 ( x 小于等于 m-2c) 用于聚合的本地clients
Krum只是找最小的一个，现在是找最小的x个
然后用Trimmed mean的一个变体来聚合这 x 个本地clients的logits。
聚合的方法是：对于每个样本，先将x个logits中的值进行排序，找到离中位数最近的 y（ y 小于等于alpha-2c）个值，取平均值；
（其实在这个规则里面，就可以看出来，恶意用户的数目c，必须要小于25%；否则这个y肯定小于0）
（对样本logits的每一维）
'''


def Bulyan(temp_all_result, args):  # 在这个规则里面，就可以看出来，恶意用户的数目c，必须要小于25%；否则这个y肯定小于0
    # clients_l2存储了，某一个client对其他client的l2范式计算
    clients_l2 = [[] for _ in range(len(temp_all_result))]

    # 求用欧几里得距离为每个client找到m-c-2个与其最近的logits
    for index1, client_logits1 in enumerate(temp_all_result):
        for index2, client_logits2 in enumerate(temp_all_result):
            if (index1 == index2):
                continue
            l2_distance = torch.dist(client_logits1, client_logits2, p=2)
            clients_l2[index1].append(l2_distance)

    clients_l2_filter = [[] for _ in range(len(temp_all_result))]
    for index, client_l2 in enumerate(clients_l2):
        list.sort(client_l2)  # 升序排列，前面的就是最小的，也就是离他最近的
        client_l2_minN = sum(
            client_l2[0:args.nbyz - 2])  # 对于单个用户client_l2，对它的前m-c-2个最近的clients求和，作为它与其他client的距离
        clients_l2_filter[index].append(client_l2_minN)

    # 在clients_l2_filter找到最小的x个用户,把他们存在selected_clients
    selected_clients = []
    x = 2 * args.nbyz - args.nworkers  # x = m - 2c = m - 2*(m-b) = 2b - m
    for i in range(x):
        selected_client_index = clients_l2_filter.index(min(clients_l2_filter))  # 找到当前的最小值；
        selected_clients.append(temp_all_result[selected_client_index])  # 添加到备选
        clients_l2_filter.pop(selected_client_index)  # 删掉这个数，相当于排除了已经被选择的client

    # 用Trimmed mean的一个变体来聚合这x个本地clients的logits
    y = x - 2 * (args.nworkers - args.nbyz)
    # 对于每个样本，m个clients都会生成对应的的logits;一共有batch个labels，用labels_logits表示
    labels_logits = [[] for _ in range(len(selected_clients[1]))]

    # 为每个label都找到最佳的logits
    for label_index1, label_logits1 in enumerate(selected_clients[0]):
        labels_temp = []
        for demission_index2, label_dimensions in enumerate(selected_clients[0][0]):
            labels_dimission_temp = []  # 记录单个label的每一维的平均
            for user_index3, client_logits in enumerate(selected_clients):
                labels_dimission_temp.append(selected_clients[user_index3][label_index1][demission_index2])
            list.sort(labels_dimission_temp)  # 排序一下
            labels_temp.append(sum(labels_dimission_temp[int((x - y) / 2):int(-(x - y) / 2)]) / len(
                labels_dimission_temp[int((x - y) / 2):int(-(x - y) / 2)]))  # 所有client针对某个label的某一维数据求平均
            # 截取找到离中位数最近的 y（ y 小于等于alpha-2c）个值, 相当于截去掉最小的(x-y)/2和最大的(x-y)/2个数, 然后求他们的平均;存到labels_logits中
        labels_logits[label_index1] = labels_temp
    agg_avg_labels = torch.Tensor(labels_logits)  # 把list转成tensor
    return agg_avg_labels


'''
对于每个样本，m个clients都会生成对应的的logits，把这些logits求其中位数，作为全局的agg_logits；
（对样本logits的每一维）
'''


def Median(temp_all_result, args):
    # agg_avg_labels = torch.median(temp_all_result, dim=0)           #在观察值为偶数的情况下，将返回两个中位数中的较低值。
    # agg_avg_labels = torch.quantile(temp_all_result,q=0.5)          #在观察值为偶数的情况下，将返回两个中位数中的平均值。

    # 对于每个样本，m个clients都会生成对应的的logits;一共有batch个labels，用labels_logits表示
    labels_logits = [[] for _ in range(len(temp_all_result[0]))]
    # 为每个label都找到最佳的logits,即用中位logits代替
    for label_index1, label_logits1 in enumerate(temp_all_result[0]):  # enumerate(temp_all_result[0])表示枚举每一个label样本
        labels_temp = []
        for demission_index2, label_dimensions in enumerate(
                temp_all_result[0][0]):  # enumerate(temp_all_result[0][0])相当于枚举每一个维度
            labels_dimission_temp = []  # 记录单个label的每一维的平均
            for user_index3, client_logits in enumerate(
                    temp_all_result):  # enumerate(temp_all_result)表示枚举每一个clients用户——相当于把每一列都取出来
                # print(len(temp_all_result[index2][index1]))
                labels_dimission_temp.append(temp_all_result[user_index3][label_index1][
                                                 demission_index2])  # temp_all_result[index2][index1][index3]索引到每个用户的每个样本的每个维度
            list.sort(labels_dimission_temp)
            # print("labels_dimission_temp:", type(labels_dimission_temp), labels_dimission_temp[beta:-beta])
            labels_temp.append(
                labels_dimission_temp[int(len(labels_dimission_temp) / 2)])  # 中位数就是长度的二分之一位置，截取之后，存到labels_logits中
        labels_logits[label_index1] = labels_temp
    agg_avg_labels = torch.Tensor(labels_logits)  # 把list转成tensor
    return agg_avg_labels


'''
对于每个样本，m个clients都会生成对应的的logits，把这些logits求平均，作为全局的agg_logits；
（对样本logits的每一维）
'''


def Mean(temp_all_result, args):
    # for index, client_logits in enumerate(temp_all_result):

    agg_avg_labels = torch.mean(temp_all_result, dim=0)  # 直接对第一维求平均，就相当于以client求平均
    # agg_avg_labels = (np.array(benign_sum_result) + np.array(poison_avg_logits.tolist()) * (
    #             args.user_number - args.benign_user_number)) / args.user_number
    # agg_avg_labels = torch.tensor(agg_avg_labels).float()
    # agg_avg_labels = agg_avg_labels.to(device)
    return agg_avg_labels


'''
Adaptive federated average（AFA）：对所有的logits求平均，
然后计算每一个clients的logits与平均logits的余弦相似度，
剔除离群值（剔除多少？百分比？20%？），
剩下的再求平均，作为全局的agg_logits；（对logits整体）
'''


def AFA(temp_all_result, args):
    # 两个向量有相同的指向时，余弦相似度的值为1；
    # 两个向量夹角为90°时，余弦相似度的值为0；
    # 两个向量指向完全相反的方向时，余弦相似度的值为-1。
    # 这结果是与向量的长度无关的，仅仅与向量的指向方向相关。
    # 余弦相似度通常用于正空间，因此给出的值为-1到1之间。
    avg_labels = torch.mean(temp_all_result, dim=0)
    attention_scores = []
    for index, client_logits in enumerate(temp_all_result):
        attention_scores.append(sum(torch.cosine_similarity(client_logits, avg_labels, dim=0)))  # dim为在哪个维度上计算余弦相似度

    # 剔除离群值
    # 先求要删除多少个数，记为 abandon_count
    abandon_count = int(len(attention_scores) * 0.2)  # 目前是剔除百分之20
    # 然后，找到attention_score中倒数第n个数——在这里是前20%的分界点；不能用排序，attention_scores的顺序不要变
    arr_min_list = heapq.nsmallest(abandon_count, attention_scores)  ##获取最小的abandon_count个值并按升序排序
    abandon_num_flag = arr_min_list[-1]  # arr_min的最后一个值就是分界点
    # 记录大于分界点的数对应的下标
    filter_index = []
    for index, attention_score in enumerate(attention_scores):
        if (attention_score > abandon_num_flag):
            filter_index.append(index)
    # 用这个下标去寻找对应的client_logits,放到filter_clients中
    print("AFA_select_index:", filter_index)
    filter_clients = []
    for index in filter_index:
        filter_clients.append(temp_all_result[index])

    # agg_avg_labels = torch.mean(torch.Tensor(filter_clients),dim=0)         #把list转换成tensor;报错“ValueError: only one element tensors can be converted to Python scalar
    agg_avg_labels = torch.mean(torch.tensor([item.cpu().detach().numpy() for item in filter_clients]).cuda(),
                                dim=0)  # 。原因是：要转换的list里面的元素包含多维的tensor。

    # agg_avg_labels = torch.mean(filter_clients,dim=0)           #把选出来的用户求平均
    return agg_avg_labels

