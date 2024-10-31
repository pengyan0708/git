import numpy as np


# 定义KL散度函数
def D1(p, q):
    return np.sum(p * np.log(p / q))


# 定义MSE函数
def D2(x, y):
    return np.mean((x - y) ** 2)


# 定义聚合函数
def Agg(l_list):
    return np.mean(l_list, axis=0)


# 定义KL散度的梯度
def gradient_D1(p, q):
    return p / q - 1


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
    penalty_value = max(0, D2(l_i_t, L_avg_t) - M * d_n_t)
    return penalty_value


# 梯度下降优化
def gradient_descent(l_i_t, L_avg_t, M, d_n_t, lambda_, learning_rate, num_iterations):
    for _ in range(num_iterations):
        L_hat = Agg(l_i_t)
        L_tilde = Agg(l_i_t[:-1])
        grad = gradient(l_i_t, lambda_, L_hat, L_tilde)
        l_i_t = l_i_t - learning_rate * grad

        # 应用惩罚以满足约束
        pen = penalty(l_i_t, L_avg_t, M, d_n_t)
        l_i_t += pen
    return l_i_t


# 牛顿法优化
def hessian_D1(p, q):
    return np.diag(p / (q ** 2))


def hessian_D2(x, y):
    return 2 * np.eye(len(x)) / len(x)


def newton_method(l_i_t, L_avg_t, M, d_n_t, lambda_, num_iterations):
    for _ in range(num_iterations):
        L_hat = Agg(l_i_t)
        L_tilde = Agg(l_i_t[:-1])
        grad = gradient(l_i_t, lambda_, L_hat, L_tilde)
        H = hessian_D1(L_hat, l_i_t) + lambda_ * hessian_D1(L_hat, L_tilde)
        l_i_t = l_i_t - np.linalg.inv(H) @ grad

        # 应用惩罚以满足约束
        pen = penalty(l_i_t, L_avg_t, M, d_n_t)
        l_i_t += pen
    return l_i_t


# 初始化变量
l_i_t = np.random.rand(10, 5)  # 示例初始化
L_avg_t = np.mean(l_i_t, axis=0)  # 示例 L_avg_t
M = 2  # 示例 M
d_n_t = 0.1  # 示例 d_n_t
lambda_ = 0.5  # 示例 lambda
learning_rate = 0.01  # 示例学习率
num_iterations = 100  # 示例迭代次数

# 运行优化
optimized_l_i_t_gd = gradient_descent(l_i_t, L_avg_t, M, d_n_t, lambda_, learning_rate, num_iterations)
print("Gradient Descent Result:")
print(optimized_l_i_t_gd)

