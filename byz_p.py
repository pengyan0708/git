from __future__ import print_function
import nd_aggregation
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
    batch_size = args.batch_size
    num_inputs, num_outputs, num_labels = data.get_shapes(args.dataset)
    byz = default.get_byz(args.byz_type)
    num_workers = args.nworkers
    lr = args.lr
    niter = args.niter

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

            else:
                raise ValueError("Invalid aggregation method selected.")

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
    if lst:
        # 绘制折线图
        plt.plot(range(len(lst)), lst)
        plt.xlabel('Iterations')
        plt.ylabel('Test Accuracy')
        plt.title(f'Test Accuracy Over Iterations\nAggregation: {args.aggregation}, Non-iid: {args.bias}, Attack Type: {args.byz_type}\nDataset: {args.dataset},Attack Number:{args.nbyz},LR:{args.lr}')
        plt.grid(True)
        plt.show()
    else:
        print("No test accuracy data available to plot.")