from __future__ import print_function
import nd_aggregation
<<<<<<< HEAD
from mxnet import nd, autograd, gluon
import numpy as np
import random
import mxnet as mx
import matplotlib.pyplot as plt
import default
import net as nt
import data



def evaluate_accuracy(data_iterator, net, ctx, trigger=False, target=None):
    # evaluate the (attack) accuracy of the model
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        remaining_idx = list(range(data.shape[0]))
        if trigger:
            data, label, remaining_idx, add_backdoor(data, label, trigger, target)
        output = net(data)
        predictions = nd.argmax(output, axis=1)                
        predictions = predictions[remaining_idx]
        label = label[remaining_idx]
        acc.update(preds=predictions, labels=label)        
    return acc.get()[1]



def main(args):
    # device to use
    ctx = default.get_device(args.gpu)
=======
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import default
import net as nt
import data,random,byzantine
import torch.nn.functional as F



def private_models_train_distillation(ctx,model,agg_images,agg_avg_labels,args):
    # SOFT_LOSE = nn.KLDivLoss(reduction="batchmean").to(device)
    SOFT_LOSE = nn.KLDivLoss(reduction='batchmean').to(ctx)
    # SOFT_LOSE = nn.L1Loss(size_average=None, reduce=None, reduction='mean').to(device)
    # SOFT_LOSE = nn.CrossEntropyLoss()
    # 蒸馏温度
    T = 10
    # alpha = 0.7
    #
    model.to(ctx)
    model.train()

    #if args.optimizer == 'sgd':
    #    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
    #                               momentum=0.5)
    #elif args.optimizer == 'adam':

    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    # criterion = nn.L1Loss(size_average=None, reduce=None, reduction='mean').to(device)
    # criterion = nn.CrossEntropyLoss()
    optimizer.zero_grad()
    agg_outputs = model(agg_images)
    agg_avg_labels = agg_avg_labels.view(32, 10)  # 将维度调整为 [32, 10]，num_outputs 是类别数
    agg_avg_labels = agg_avg_labels.to(agg_outputs.device)
    #print(agg_outputs.shape)
    #print(agg_avg_labels.shape)
    # loss = criterion(outputs, agg_avg_labels)  # 算新的交叉熵，第一种，直接全部用教师模型的输出去指导学生，没有hard_labels
    # hard_loss_private = HARD_LOSE(local_outputs, local_labels)  # labels就是原始的(images, labels) 集合
    # ditillation_loss = soft_loss(F.softmax(out / T, dim=1), F.softmax(teacher_output / T, dim=1))
    ditillation_loss_private = SOFT_LOSE(F.log_softmax(agg_outputs / T, dim=1),F.softmax(agg_avg_labels / T, dim=1))  # agg_avg_labels现在就是联合模型的综合输出，用于指导模型
    # loss_all = hard_loss_private * alpha + ditillation_loss_private * (1 - alpha)
    loss_all = ditillation_loss_private
    loss_all.backward()
    # 这里需要调整一下，知识蒸馏应该是hard_labels+soft_labels的混合；这里似乎只有soft_labels
    # loss.backward()
    optimizer.step()
    return loss_all


def evaluate_accuracy(data_loader, net, ctx, trigger=False, target=None):
    # evaluate the (attack) accuracy of the model
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, label in data_loader:
            data, label = data.to(ctx), label.to(ctx)
            output = net(data)
            _, predicted = torch.max(output, 1)
            correct += (predicted == label).sum().item()
            total += label.size(0)
    return correct / total


def custom_loss_function(L_i, L_hat_i, L_tilde_i, ctx, model, args, lambda_param=1.0):
    model.to(ctx)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    optimizer.zero_grad()
    L_i = L_i.view(320, 1).clone().detach().requires_grad_(True)
    L_hat_i = L_hat_i.clone().detach().requires_grad_(True)
    L_tilde_i = L_tilde_i.clone().detach().requires_grad_(True)
    kl_loss_1 = kl_divergence(L_hat_i, L_i)
    kl_loss_2 = kl_divergence(L_hat_i, L_tilde_i)
    loss_all = kl_loss_1 - lambda_param * kl_loss_2
    loss_all.backward()
    optimizer.step()
    return loss_all


def main(args):
    # set device
    ctx = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
>>>>>>> 5f3ea6c98597f0f3016e5a929eb8a36fc9302929
    batch_size = args.batch_size
    num_inputs, num_outputs, num_labels = data.get_shapes(args.dataset)
    byz = default.get_byz(args.byz_type)
    num_workers = args.nworkers
    lr = args.lr
    niter = args.niter

<<<<<<< HEAD
    paraString = 'p' + str(args.p) + '_' + str(args.dataset) + "server " + str(args.server_pc) + "bias" + str(
        args.bias) + "+nworkers " + str(
        args.nworkers) + "+" + "net " + str(args.net) + "+" + "niter " + str(args.niter) + "+" + "lr " + str(
        args.lr) + "+" + "batch_size " + str(args.batch_size) + "+nbyz " + str(
        args.nbyz) + "+" + "byz_type " + str(args.byz_type) + "+" + "aggregation " + str(args.aggregation) + ".txt"

    with ctx:

        # model architecture
        net = nt.get_net(args.net, num_outputs)
        # initialization
        net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), force_reinit=True, ctx=ctx)
        # loss
        softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

        grad_list = []
        test_acc_list = []

        # load the data
        # fix the seeds for loading data
        seed = args.nrepeats
        if seed > 0:
            mx.random.seed(seed)
            random.seed(seed)
            np.random.seed(seed)
        train_data, test_data = data.load_data(args.dataset)

        # assign data to the server and clients
        server_data, server_label, each_worker_data, each_worker_label = data.assign_data(
            train_data, args.bias, ctx, num_labels=num_labels, num_workers=num_workers,
            server_pc=args.server_pc, p=args.p, dataset=args.dataset, seed=seed)

        # begin training
        for e in range(niter):
            total_loss = 0.0  # 记录总损失值
            for i in range(num_workers):
                minibatch = np.random.choice(list(range(each_worker_data[i].shape[0])), size=batch_size, replace=False)
                with autograd.record():
                    output = net(each_worker_data[i][minibatch])
                    loss = softmax_cross_entropy(output, each_worker_label[i][minibatch])
                loss.backward()
                grad_list.append([param.grad().copy() for param in net.collect_params().values()])
                total_loss += loss.mean().asscalar()  # 累积每个客户端的损失值

            if args.aggregation == "fltrust":
                # compute server update and append it to the end of the list
                minibatch = np.random.choice(list(range(server_data.shape[0])), size=args.server_pc, replace=False)
                with autograd.record():
                    output = net(server_data)
                    loss = softmax_cross_entropy(output, server_label)
                loss.backward()
                grad_list.append([param.grad().copy() for param in net.collect_params().values()])
                total_loss += loss.mean().asscalar()  # 累积服务器的损失值
                # perform the aggregation
                nd_aggregation.fltrust(grad_list, net, lr, args.nbyz, byz)

            elif args.aggregation == "fed_avg":
                nd_aggregation.fed_avg(grad_list, net, lr, args.nbyz, byz)

            elif args.aggregation == "fed_med":
                nd_aggregation.fed_med(grad_list, net, lr, args.nbyz, byz)

            elif args.aggregation == "krum":
                nd_aggregation.krum(grad_list, net, lr, args.nbyz, byz)

            elif args.aggregation == "trimmed_mean":
                nd_aggregation.trimmed_mean(grad_list, net, lr, args.nbyz, byz)
=======
    paraString = f'p{args.p}_{args.dataset}server {args.server_pc}bias{args.bias}+nworkers {args.nworkers}+' \
                 f'net {args.net}+niter {args.niter}+lr {args.lr}+batch_size {args.batch_size}+nbyz {args.nbyz}+' \
                 f'byz_type {args.byz_type}+aggregation {args.aggregation}.txt'

    # load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset, test_dataset = data.load_data(args.dataset)

    # assign data to the server and clients
    server_data, server_label, each_worker_data, each_worker_label = data.assign_data(
        train_dataset, args.bias, ctx, num_labels=num_labels, num_workers=num_workers,
        server_pc=args.server_pc, p=args.p, dataset=args.dataset, seed=args.nrepeats)


    # model architecture
    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

    client_net = []
    net_name = ['cnn','cnn1','cnn2','cnn3','cnn4']
    for idx in range(num_workers):
        # 根据需要选择网络名称，这里假设每个client都使用'cnn_unique'作为示例
        # 但实际上你可以为每个client提供一个不同的网络名称或配置
        client_net.append(nt.get_net(net_name[idx], num_outputs).to(ctx))

    for i in range(num_workers):
    # initialization
        client_net[i].apply(init_weights)
    # loss
    criterion = nn.CrossEntropyLoss()
    optimizers = [optim.SGD(client_net[i].parameters(), lr=lr) for i in range(num_workers)]

    grad_list = []
    logits_list = []
    client_accuracy_dict = {}
    temp_all_result = None
    test_acc_list = []
    non_self_test_acc_list = []
    self_test_acc_list = []




    # begin training
    for e in range(niter):
        total_loss = 0.0
        data_batch_list = []
        for i in range(num_workers):
            client_net[i].train()
            indices = np.random.choice(len(each_worker_data[i]), batch_size, replace=False)
            data_batch = each_worker_data[i][indices].clone().detach().to(torch.float32).to(ctx)
            label_batch = each_worker_label[i][indices].clone().detach().to(torch.float32).to(ctx)
            data_batch_list.append(data_batch)
            optimizers[i].zero_grad()
            output = client_net[i](data_batch)
            label_batch = label_batch.long()  # 将label_batch转换为torch.long类型
            loss = criterion(output, label_batch)
            loss.backward(retain_graph=True)
            grad_list.append([param.grad.clone() for param in client_net[i].parameters()])
            logits_list.append(output.clone().unsqueeze(0))  # 添加模型输出 logits
            #if (temp_all_result == None):
                #temp_all_result = output.clone().unsqueeze(0)
            #else:
                # temp_all_result =  torch.stack((temp_all_result,outputs.clone()),dim=0)     #stack是堆积，相当于开创了一个人数维度；问题是第二次拼接的时候temp_all_result比outputs多了一个维度
                # 考虑用unsqueeze先给outputs增加一个维度，然后按新维度cat
                #temp_all_result = torch.cat((temp_all_result, output.clone().unsqueeze(0)), dim=0)
            optimizers[i].step()
            total_loss += loss.item()

        for i in range(num_workers):
            if args.aggregation == "fltrust":
                # compute server update and append it to the end of the list
                indices = np.random.choice(len(server_data), args.server_pc, replace=False)
                # server_data_batch = torch.tensor(server_data[indices], dtype=torch.float32, device=ctx)
                # server_label_batch = torch.tensor(server_label[indices], dtype=torch.long, device=ctx)
                # optimizers[i].zero_grad()
                # server_output = net(server_data_batch)
                # server_label_batch = server_label_batch.long()
                # server_loss = criterion(server_output, server_label_batch)
                # server_loss.backward()
                # grad_list.append([param.grad.clone() for param in net.parameters()])
                # logits_list.append(server_output.detach().cpu().numpy())  # 添加模型输出 logits
                # optimizer.step()
                # total_loss += server_loss.item()
                # nd_aggregation.fltrust(grad_list, net, lr, args.nbyz, byz)

            elif args.aggregation == "fed_avg":
                if i >= args.nbyz:
                    logits_list = byz(logits_list,args)
                temp_all_result,non_avg_result = nd_aggregation.fed_avg(logits_list)
                data_batch = data_batch_list[i]
                local_loss = private_models_train_distillation(ctx, client_net[i], data_batch, temp_all_result, args)
                # op_loss = custom_loss_function(logits_list[i],temp_all_result,non_avg_result,ctx, client_net[i],args)
                print(f'第{e + 1}轮____第{i + 1}客户端蒸馏之后loss：{local_loss.item()}')
                # print(f'第{e + 1}轮____第{i + 1}客户端loss：{op_loss.item()}')


            elif args.aggregation == "fed_med":
                temp_all_result = nd_aggregation.fed_med(logits_list, client_net[i],i+1,lr, args.nbyz, byz)
                #temp_all_result.to(ctx)
                data_batch = data_batch_list[i]
                local_loss = private_models_train_distillation(ctx,client_net[i],data_batch,temp_all_result,args)
                print(f'第{e + 1}轮____第{i + 1}客户端蒸馏之后loss：{local_loss.item()}')

            elif args.aggregation == "krum":
                temp_all_result = nd_aggregation.krum(logits_list, client_net[i],i+1,lr, args.nbyz, byz)
                data_batch = data_batch_list[i]
                local_loss = private_models_train_distillation(ctx, client_net[i], data_batch, temp_all_result, args)
                print(f'第{e + 1}轮____第{i + 1}客户端蒸馏之后loss：{local_loss.item()}')



            elif args.aggregation == "trimmed_mean":
                temp_all_result = nd_aggregation.trimmed_mean(logits_list, client_net[i],i+1,lr, args.nbyz, byz)
                data_batch = data_batch_list[i]
                local_loss = private_models_train_distillation(ctx, client_net[i], data_batch, temp_all_result, args)
                print(f'第{e + 1}轮____第{i + 1}客户端蒸馏之后loss：{local_loss.item()}')

>>>>>>> 5f3ea6c98597f0f3016e5a929eb8a36fc9302929

            else:
                raise ValueError("Invalid aggregation method selected.")

<<<<<<< HEAD
            del grad_list
            grad_list = []

            # evaluate the model accuracy
            test_accuracy = evaluate_accuracy(test_data, net, ctx)
            test_acc_list.append(test_accuracy)
            # 计算平均损失
            avg_loss = total_loss / num_workers
            print("Iteration %02d Test_acc %0.4f Loss: %0.4f" % (e+1, test_accuracy,avg_loss))

        return test_acc_list

if __name__ == "__main__":
    args = default.parse_args("MNIST",0,"krum_attack","krum")
    lst = main(args)
    print(lst)
=======
        del grad_list
        grad_list = []

        # 初始化字典存储每个客户端的准确度数组
        test_accuracy = 0
        self_accuracy = 0
        # 遍历每个客户端
        for i in range(num_workers):
            # 计算当前客户端的测试准确度
            client_accuracy = evaluate_accuracy(test_dataset, client_net[i], ctx)

            print("第 %02d 轮 第 %02d 个客户端 Test_acc %0.4f" % (e + 1, i + 1, client_accuracy))
            test_accuracy += client_accuracy
            if i<2:
                self_accuracy += client_accuracy
            # 将准确度存储到对应客户端的数组中
            if i not in client_accuracy_dict:
                client_accuracy_dict[i] = []  # 如果客户端还没有对应的准确度数组，则初始化一个空数组
            client_accuracy_dict[i].append(client_accuracy)  # 存储当前轮次的准确度到数组中

        non_self_accuracy = (test_accuracy - self_accuracy) / 3
        test_accuracy = test_accuracy / num_workers
        self_accuracy = self_accuracy / 2

        test_acc_list.append(test_accuracy)
        non_self_test_acc_list.append(non_self_accuracy)
        self_test_acc_list.append(self_accuracy)
        # 打印字典中存储的每个客户端的准确度数组
        # for i in range(num_workers):
        #     print(f"客户端 {i + 1} 的准确度数组: {client_accuracy_dict[i]}")



        # 计算平均损失
        avg_loss = total_loss / num_workers
        print("Iteration %02d 自私用户准确度： %0.4f  非自私用户准确度： %0.4f" % (e+1, self_accuracy,non_self_accuracy))
        print("Iteration %02d Test_acc %0.4f Loss: %0.4f" % (e+1, test_accuracy,avg_loss))

    return test_acc_list,self_test_acc_list,non_self_test_acc_list

if __name__ == "__main__":
    args = default.parse_args("FashionMNIST",0,"trim_attack","fed_avg")
    lst,self_lst,non_self_lst = main(args)
    print("总体准确度数组：")
    print(lst)
    print("自私用户准确度数组：")
    print(self_lst)
    print("非自私用户准确度数组：")
    print(non_self_lst)
>>>>>>> 5f3ea6c98597f0f3016e5a929eb8a36fc9302929
    if lst:
        # 绘制折线图
        plt.plot(range(len(lst)), lst)
        plt.xlabel('Iterations')
        plt.ylabel('Test Accuracy')
        plt.title(f'Test Accuracy Over Iterations\nAggregation: {args.aggregation}, Non-iid: {args.bias}, Attack Type: {args.byz_type}\nDataset: {args.dataset},Attack Number:{args.nbyz},LR:{args.lr}')
        plt.grid(True)
        plt.show()
    else:
<<<<<<< HEAD
=======
        print("No test accuracy data available to plot.")

    if non_self_lst or self_lst:
        plt.plot(range(len(self_lst)), self_lst, label='self_acc')
        plt.plot(range(len(non_self_lst)), non_self_lst, label='non_self_acc')
        # plt.title(f'Test Accuracy Over Iterations\nAggregation: {lst["aggregation"]}, Non-iid: {lst["bias"]}, Attack Type: {lst["byz_type"]}\nDataset: {lst["dataset"]},Attack Number:{lst["nbyz"]},LR:{lst["lr"]}')
        plt.xlabel('Iterations')
        plt.ylabel('Test Accuracy')
        plt.title(f'Test Accuracy Over Iterations\nAggregation: {args.aggregation}, Non-iid: {args.bias}, Attack Type: {args.byz_type}\nDataset: {args.dataset},Attack Number:{args.nbyz},LR:{args.lr}')
        plt.grid(True)
        # 显示图例
        plt.legend()
        # 显示图
        plt.show()
    else:
>>>>>>> 5f3ea6c98597f0f3016e5a929eb8a36fc9302929
        print("No test accuracy data available to plot.")