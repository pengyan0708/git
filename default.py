import argparse
import torch
import byzantine

def parse_args(dataset, bias, byz_type, aggregation):
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_pc", help="the number of data the server holds", type=int, default=5)
    parser.add_argument("--dataset", help="dataset", type=str, default=dataset)
    parser.add_argument("--bias", help="degree of non-iid", type=float, default=bias)
    parser.add_argument("--net", help="net", type=str, default="cnn")
    parser.add_argument("--batch_size", help="batch size", type=int, default=32)
    parser.add_argument("--lr", help="learning rate", type=float, default=0.1)
    parser.add_argument("--nworkers", help="# workers", type=int, default=5)
    parser.add_argument("--niter", help="# iterations", type=int, default=300)
    parser.add_argument("--gpu", help="index of gpu", type=int, default=0)
    parser.add_argument("--nrepeats", help="seed", type=int, default=0)
    parser.add_argument("--nbyz", help="# byzantines", type=int, default=2)
    parser.add_argument("--byz_type", help="type of attack", type=str, default=byz_type)
    parser.add_argument("--aggregation", help="aggregation", type=str, default=aggregation)
    parser.add_argument("--p", help="bias probability of 1 in server sample", type=float, default=0.1)
    args = parser.parse_args()

    return args

def get_device(device):
    # define the device to use
    if device == -1:
        ctx = torch.device('cpu')
    else:
        ctx = torch.device('cuda', device)
    return ctx

def get_byz(byz_type):
    # get the attack type
    if byz_type == "no":
        return byzantine.no_byz
    elif byz_type == 'trim_attack':
        return byzantine.generate_trimmed_mean_fake_logits
    elif byz_type == 'krum_attack':
        return byzantine.generate_krum_fake_logits
    else:
        raise NotImplementedError
