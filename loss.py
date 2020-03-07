# list all the additional loss functions

import torch
import torch.nn as nn


# entropy loss (continuous target)
def cross_entropy_soft(pred):
    softmax = nn.Softmax(dim=1)
    logsoftmax = nn.LogSoftmax(dim=1)
    loss = torch.mean(torch.sum(-softmax(pred) * logsoftmax(pred), 1))
    return loss


# attentive entropy loss (source + target)
def attentive_entropy(pred, pred_domain):
    softmax = nn.Softmax(dim=1)
    logsoftmax = nn.LogSoftmax(dim=1)

    # attention weight
    entropy = torch.sum(-softmax(pred_domain) * logsoftmax(pred_domain), 1)
    weights = 1 + entropy

    # attentive entropy
    loss = weights * torch.sum(-softmax(pred) * logsoftmax(pred), 1)
    return loss


# discrepancy for ensemble loss
def dis_mcd(out1, out2):
    return torch.mean(torch.abs(out1 - out2))


def dis_swd(p1, p2, dim_proj=128):
    s = p1.shape
    if s[1] > 1:
        proj = torch.randn(s[1], dim_proj)
        if p1.get_device() >= 0:
            proj = proj.to(p1.get_device())
        proj *= torch.rsqrt(torch.sum(torch.mul(proj, proj), 0, keepdim=True))
        p1 = torch.matmul(p1, proj)
        p2 = torch.matmul(p2, proj)
    p1 = torch.topk(p1, s[0], dim=0)[0]
    p2 = torch.topk(p2, s[0], dim=0)[0]
    dist = p1 - p2
    wdist = torch.mean(torch.mul(dist, dist))

    return wdist


# discrepancy loss
def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    dist_l2 = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(dist_l2.detach()) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-dist_l2 / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)


def loss_jan(source_list, target_list, kernel_muls=[2.0, 2.0], kernel_nums=[5, 1], fix_sigma_list=[None, 1.68], ver=2):
    batch_size = int(source_list[0].size()[0])
    layer_num = len(source_list)
    joint_kernels = None
    for i in range(layer_num):
        source = source_list[i]
        target = target_list[i]
        kernel_mul = kernel_muls[i]
        kernel_num = kernel_nums[i]
        fix_sigma = fix_sigma_list[i]
        kernels = gaussian_kernel(source, target,
                                  kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        if joint_kernels is not None:
            joint_kernels = joint_kernels * kernels
        else:
            joint_kernels = kernels

    loss = 0

    if ver == 1:
        for i in range(batch_size):
            s1, s2 = i, (i + 1) % batch_size
            t1, t2 = s1 + batch_size, s2 + batch_size
            loss += joint_kernels[s1, s2] + joint_kernels[t1, t2]
            loss -= joint_kernels[s1, t2] + joint_kernels[s2, t1]
        loss = loss.abs() / float(batch_size)
    elif ver == 2:
        xx = joint_kernels[:batch_size, :batch_size]
        yy = joint_kernels[batch_size:, batch_size:]
        xy = joint_kernels[:batch_size, batch_size:]
        yx = joint_kernels[batch_size:, :batch_size]
        loss = torch.mean(xx + yy - xy - yx)
    else:
        raise ValueError('ver == 1 or 2')

    return loss
