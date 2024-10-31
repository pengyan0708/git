import mxnet as mx
from mxnet import nd, autograd, gluon
import numpy as np

def fltrust(gradients, net, lr, f, byz):
    """
    gradients: list of gradients. The last one is the server update.
    net: model parameters.
    lr: learning rate.
    f: number of malicious clients. The first f clients are malicious.
    byz: attack type.
    """
    
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    # let the malicious clients (first f clients) perform the byzantine attack
    param_list = byz(param_list, net, lr, f)
    n = len(param_list) - 1
    
    # use the last gradient (server update) as the trusted source
    baseline = nd.array(param_list[-1]).squeeze()
    cos_sim = []
    new_param_list = []
    
    # compute cos similarity
    for each_param_list in param_list:
        each_param_array = nd.array(each_param_list).squeeze()
        cos_sim.append(nd.dot(baseline, each_param_array) / (nd.norm(baseline) + 1e-9) / (nd.norm(each_param_array) + 1e-9))

        
    cos_sim = nd.stack(*cos_sim)[:-1]
    cos_sim = nd.maximum(cos_sim, 0) # relu
    normalized_weights = cos_sim / (nd.sum(cos_sim) + 1e-9) # weighted trust score

    # normalize the magnitudes and weight by the trust score
    for i in range(n):
        new_param_list.append(param_list[i] * normalized_weights[i] / (nd.norm(param_list[i]) + 1e-9) * nd.norm(baseline))
    
    # update the global model
    global_update = nd.sum(nd.concat(*new_param_list, dim=1), axis=-1)

    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        param.set_data(param.data() - lr * global_update[idx:(idx+param.data().size)].reshape(param.data().shape))
        idx += param.data().size       

def fed_avg(gradients, net, lr, f, byz):
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    param_list = byz(param_list, net, lr, f)

    # 计算所有梯度的平均值
    avg_gradient = nd.mean(
        nd.stack(*param_list), axis=0)

    # 更新全局模型
    idx = 0
    for param in net.collect_params().values():
        param.set_data(param.data() - lr * avg_gradient[idx:(idx + param.data().size)].reshape(param.data().shape))
        idx += param.data().size

def fed_med(gradients, net, lr, f, byz):
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    param_list = byz(param_list, net, lr, f)

    # 计算所有梯度的中位数
    median_gradient = np.median(nd.concat(*param_list, dim=1).asnumpy(), axis=1)
    median_gradient = nd.array(median_gradient, ctx=param_list[0].context).reshape((-1, 1))

    # 更新全局模型
    idx = 0
    for param in net.collect_params().values():
        param.set_data(param.data() - lr * median_gradient[idx:(idx + param.data().size)].reshape(param.data().shape))
        idx += param.data().size


def krum(gradients, net, lr, f, byz, k=3):
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    param_list = byz(param_list, net, lr, f)

    distances = []
    for i in range(len(param_list)):
        dist = sum(nd.norm(param_list[i] - param_list[j]) for j in range(len(param_list)) if i != j)
        distances.append(dist.asscalar())
    indices = np.argsort(distances)[:k]
    krum_gradient = sum(param_list[i] for i in indices) / k

    # 更新全局模型
    idx = 0
    for param in net.collect_params().values():
        param.set_data(param.data() - lr * krum_gradient[idx:(idx + param.data().size)].reshape(param.data().shape))
        idx += param.data().size


def trimmed_mean(gradients, net, lr, f, byz, trim_ratio=0.1):
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    param_list = byz(param_list, net, lr, f)

    num_to_keep = int(len(param_list) * (1 - trim_ratio))
    sorted_gradients = sorted(param_list, key=lambda x: nd.norm(x).asscalar())
    trimmed_gradients = sorted_gradients[:num_to_keep]
    trimmed_mean_gradient = sum(trimmed_gradients) / num_to_keep

    # 更新全局模型
    idx = 0
    for param in net.collect_params().values():
        param.set_data(param.data() - lr * trimmed_mean_gradient[idx:(idx + param.data().size)].reshape(param.data().shape))
        idx += param.data().size







