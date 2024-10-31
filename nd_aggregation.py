import torch
import numpy as np

def fltrust(gradients, net, lr, f, byz):
    """
    gradients: list of gradients. The last one is the server update.
    net: model parameters.
    lr: learning rate.
    f: number of malicious clients. The first f clients are malicious.
    byz: attack type.
    """

    param_list = [torch.cat([xx.view(-1, 1) for xx in x], dim=0) for x in gradients]
    # let the malicious clients (first f clients) perform the byzantine attack
    param_list = byz(param_list, net, lr, f)
    n = len(param_list) - 1

    # use the last gradient (server update) as the trusted source
    baseline = torch.tensor(param_list[-1]).squeeze()
    cos_sim = []

    # compute cos similarity
    for each_param_list in param_list:
        each_param_array = torch.tensor(each_param_list).squeeze()
        cos_sim.append(torch.dot(baseline, each_param_array) / (torch.norm(baseline) + 1e-9) / (torch.norm(each_param_array) + 1e-9))

    cos_sim = torch.stack(cos_sim)[:-1]
    cos_sim = torch.clamp(cos_sim, min=0)  # relu
    normalized_weights = cos_sim / (torch.sum(cos_sim) + 1e-9)  # weighted trust score

    new_param_list = []

    # normalize the magnitudes and weight by the trust score
    for i in range(n):
        new_param_list.append(param_list[i] * normalized_weights[i] / (torch.norm(param_list[i]) + 1e-9) * torch.norm(baseline))

    # update the global model
    global_update = torch.sum(torch.cat(new_param_list, dim=1), dim=-1)

    #idx = 0
    #for param in net.parameters():
    #    param.data -= lr * global_update[idx:(idx + param.data.numel())].view(param.data.shape)
    #    idx += param.data.numel()
    return global_update

def fed_avg(gradients):
    param_list = [torch.cat([xx.view(-1, 1) for xx in x], dim=0) for x in gradients]
    #if m > f:
    #    param_list = byz(param_list, f)
    non_param_list = [torch.cat([xx.view(-1, 1) for xx in x], dim=0) for x in gradients[-3:]]


    # compute the average of all gradients
    avg_gradient = torch.mean(torch.stack(param_list), dim=0)
    avg_gradient = torch.tensor(avg_gradient, device=param_list[0].device).clone().detach().view(-1, 1)
    non_avg_logits = torch.mean(torch.stack(non_param_list), dim=0)
    non_avg_logits = torch.tensor(avg_gradient, device=non_avg_logits[0].device).clone().detach().view(-1, 1)

    return avg_gradient,non_avg_logits

def fed_med(logits, net, m, lr, f, byz):
    param_list = [torch.cat([xx.view(-1, 1) for xx in x], dim=0) for x in logits]
    param_list = byz(param_list, net, lr, f)


    median_logits = torch.median(torch.cat(param_list, dim=1), dim=1)[0].detach().cpu().numpy()
    median_logits = torch.tensor(median_logits, device=param_list[0].device).view(-1, 1)


    return median_logits

def krum(gradients, net, m, lr, f, byz, k=3):
    param_list = [torch.cat([xx.view(-1, 1) for xx in x], dim=0) for x in gradients]
    if m > f:
        param_list = byz(param_list, f)

    distances = []
    for i in range(len(param_list)):
        dist = sum(torch.norm(param_list[i] - param_list[j]) for j in range(len(param_list)) if i != j)
        distances.append(dist.item())
    indices = np.argsort(distances)[:k]
    krum_gradient = sum(param_list[i] for i in indices) / k
    krum_gradient = torch.tensor(krum_gradient, device=param_list[0].device).clone().detach().view(-1, 1)


    return krum_gradient

def trimmed_mean(gradients,  trim_ratio=0.1):
    param_list = [torch.cat([xx.view(-1, 1) for xx in x], dim=0) for x in gradients]

    num_to_keep = int(len(param_list) * (1 - trim_ratio))
    sorted_gradients = sorted(param_list, key=lambda x: torch.norm(x).item())
    trimmed_gradients = sorted_gradients[:num_to_keep]
    trimmed_mean_gradient = sum(trimmed_gradients) / num_to_keep
    trimmed_mean_gradient = torch.tensor(trimmed_mean_gradient, device=param_list[0].device).clone().detach().view(-1, 1)

    return trimmed_mean_gradient
