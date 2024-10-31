import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import nd_aggregation


# 定义KL散度函数
def D1(p, q):
    return torch.sum(p * torch.log(p / (q + 1e-10)))  # 避免对零取对数


# 定义MSE函数
def D2(x, y):
    return torch.mean((x - y) ** 2)


# 定义聚合函数
def Agg(l_list, args):
    if args.aggregation == "trimmed_mean":
        return nd_aggregation.trimmed_mean(torch.stack(l_list))
    return torch.mean(torch.stack(l_list), dim=0)


# 定义KL散度的梯度
def gradient_D1(p, q):
    return p / (q + 1e-10) - 1  # 避免除以零


# 定义MSE的梯度
def gradient_D2(x, y):
    return 2 * (x - y) / len(x)


# 目标函数和梯度
def objective(l_i_t, lambda_, L_hat, L_tilde):
    return D1(L_hat, l_i_t) - lambda_ * D1(L_hat, L_tilde)


def gradient(l_i_t, lambda_, L_hat, L_tilde):
    grad_D1 = gradient_D1(L_hat, l_i_t) - lambda_ * gradient_D1(L_hat, L_tilde)
    return grad_D1


# 约束惩罚
def penalty(l_i_t, L_avg_t, M, d_n_t):
    penalty_value = torch.clamp(D2(l_i_t, L_avg_t) - M * d_n_t, min=0)  # 避免负值
    return penalty_value


# 梯度下降优化
def gradient_descent(l_i_t, logits, L_avg_t, M, d_n_t,args, lambda_, learning_rate, num_iterations):
    for i in range(num_iterations):
        L_hat = Agg(logits, args)
        L_tilde = Agg(logits[M:], args)
        grad = gradient(l_i_t, lambda_, L_hat, L_tilde)

        # 打印中间梯度信息以进行调试
        #print(f"Iteration {i}: Gradient - {grad}")

        # 应用梯度裁剪避免梯度爆炸
        torch.nn.utils.clip_grad_norm_([l_i_t], max_norm=1.0)

        # 更新参数
        l_i_t = l_i_t - learning_rate * grad

        # 应用惩罚以满足约束
        pen = penalty(l_i_t, L_avg_t, M, d_n_t)
        l_i_t += pen

        # 打印中间结果以进行调试
        #print(f"Iteration {i}: l_i_t - {l_i_t}")

        # 检查是否出现nan或inf
        if torch.isnan(l_i_t).any() or torch.isinf(l_i_t).any():
            print(f"Iteration {i}: NaN or Inf detected in l_i_t")
            break

    return l_i_t



def no_byz(v, net, lr, f):
    return v

def trim_attack(v, net, lr, f):
    # local model poisoning attack against Trimmed-mean
    vi_shape = v[0].shape
    v_tran = torch.cat(v, dim=1)
    maximum_dim, _ = torch.max(v_tran, dim=1, keepdim=True)
    minimum_dim, _ = torch.min(v_tran, dim=1, keepdim=True)
    direction = torch.sign(torch.sum(v_tran, dim=1, keepdim=True))
    directed_dim = (direction > 0) * minimum_dim + (direction < 0) * maximum_dim

    # let the malicious clients (first f clients) perform the attack
    for i in range(f):
        random_12 = 1. + torch.rand(vi_shape).to(v[i].device)  # 将随机张量移动到v[i]的设备上
        v[i] = directed_dim * ((direction * directed_dim > 0) / random_12 + (direction * directed_dim < 0) * random_12)

    return v

def generate_krum_fake_logits(logits, num_byzantine, epsilon=0.1):
    logits = logits[num_byzantine:]
    normal_mean = torch.mean(torch.stack(logits), dim=0)
    normal_std = torch.std(torch.stack(logits), dim=0)
    fake_logits = []
    for _ in range(num_byzantine):
        noise = torch.randn_like(normal_mean) * epsilon
        fake_logit = normal_mean + noise
        fake_logits.append(fake_logit)
    logits = fake_logits + logits
    return logits

def generate_trimmed_mean_fake_logits(logits, args):
    logits1 = logits[args.nbyz:]
    L_avg_t = torch.mean(torch.stack(logits1), dim=0)
    d_n_t = torch.max(torch.tensor([D2(logits1[i].cpu(), L_avg_t.cpu()) for i in range(len(logits1))]))
    fake_logits = []
    for _ in range(args.nbyz):
        fake_logit = gradient_descent(logits[_],logits,L_avg_t,args.nbyz,d_n_t,args,0.5,0.0001,0)
        #print(fake_logit)
        fake_logits.append(fake_logit)
    logits = fake_logits + logits1
    return logits


def krum_attack(v, net, lr, f):
    vi_shape = v[0].shape
    v_tran = torch.cat(v, dim=1)
    direction = torch.sign(torch.sum(v_tran, dim=1, keepdim=True))

    # 获取前一次的全局模型参数
    Wre = torch.cat([param.data.view(-1, 1) for param in net.parameters()], dim=0)

    # 初始化上界参数 l
    l = 5.0  # 增大初始上界
    threshold = 1e-5  # 设定阈值

    while l >= threshold:
        for i in range(f):
            modified_gradient = Wre + l * direction
            v[i] = modified_gradient.view(*vi_shape)

        # 将所有梯度合并为一个大的 Tensor，然后计算均值
        all_gradients = torch.cat(v, dim=0)
        mean_gradient = torch.mean(all_gradients, dim=0)
        # 计算每个梯度与均值的欧氏距离
        distances = [torch.norm(grad - mean_gradient) for grad in v]

        # 找到距离最小的 K-1 个梯度的索引
        num_to_keep = len(v) - 1
        indices = np.argsort([dist.item() for dist in distances])[:num_to_keep]

        if 0 in indices:  # 如果第一个被修改的本地模型被选中，则说明当前 l 是一个可行解
            break
        else:
            l /= 2.0  # 否则将 l 减半继续尝试

    return v
