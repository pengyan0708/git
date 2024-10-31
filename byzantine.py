import numpy as np
from mxnet import nd, autograd, gluon

def no_byz(v, net, lr, f):
    return v

def trim_attack(v, net, lr, f):
    # local model poisoning attack against Trimmed-mean
    vi_shape = v[0].shape
    v_tran = nd.concat(*v, dim=1)
    maximum_dim = nd.max(v_tran, axis=1).reshape(vi_shape)
    minimum_dim = nd.min(v_tran, axis=1).reshape(vi_shape)
    direction = nd.sign(nd.sum(nd.concat(*v, dim=1), axis=-1, keepdims=True))
    directed_dim = (direction > 0) * minimum_dim + (direction < 0) * maximum_dim
    # let the malicious clients (first f clients) perform the attack
    for i in range(f):
        random_12 = 1. + nd.random.uniform(shape=vi_shape)
        v[i] = directed_dim * ((direction * directed_dim > 0) / random_12 + (direction * directed_dim < 0) * random_12)
    return v



def krum_attack(v, net, lr, f):
    vi_shape = v[0].shape
    v_tran = nd.concat(*v, dim=1)
    direction = nd.sign(nd.sum(v_tran, axis=1, keepdims=True))  # 计算梯度变化的方向

    # 获取前一次的全局模型参数
    Wre = nd.concat(*[param.data().reshape((-1, 1)) for param in net.collect_params().values()], dim=0)

    # 初始化上界参数 l
    l = 5  # 增大初始上界
    threshold = 1e-5  # 设定阈值

    while l >= threshold:
        for i in range(f):
            modified_gradient = Wre + l * direction
            v[i] = modified_gradient

        # 将所有梯度合并为一个大的 NDArray，然后计算均值
        all_gradients = nd.concat(*v, dim=0)
        mean_gradient = nd.mean(all_gradients, axis=0)

        # 计算每个梯度与均值的欧氏距离
        distances = [nd.norm(grad - mean_gradient) for grad in v]

        # 找到距离最小的 K-1 个梯度的索引
        num_to_keep = len(v) - 1
        indices = np.argsort(distances)[:num_to_keep]

        if 0 in indices:  # 如果第一个被修改的本地模型被选中，则说明当前 l 是一个可行解
            break
        else:
            l /= 2.0  # 否则将 l 减半继续尝试

    return v





