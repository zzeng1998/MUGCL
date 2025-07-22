import torch
import torch.nn.functional as F
#device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
import numpy as np

def SimCSE_loss(pred, device, tau=0.05):
    ids = torch.arange(0, pred.shape[0], device=device)
    y_true=ids+1-ids%2*2
    similarities = F.cosine_similarity(pred.unsqueeze(1), pred.unsqueeze(0), dim=2)
    #屏蔽对角矩阵，即自身相等的loss
    similarities = similarities - torch.eye(pred.shape[0], device=device) * 1e12
    similarities = similarities / tau
    return torch.mean(F.cross_entropy(similarities, y_true))
pred = torch.tensor([[0.1, 0.1, 0.1, 0.1],[0.3, 0.2, 0.2, 0.2],[0.9, 0.3, 0.3, 0.3],[0.9, 0.4, 0.4, 0.4],[0.5,0.8,0.9,0.8],[0.5,0.8,0.9,0.8]])
blank_state = torch.tensor([[0.5, 0.1, 0.1, 0.1],[0.7, 0.2, 0.2, 0.2],[0.8, 0.3, 0.3, 0.3],[0.6, 0.4, 0.4, 0.4]])
#pred = pred.to(device)
#blank_state = blank_state.to(device)
#print(SimCSE_loss(pred))

def simcse_unsup_loss(y_pred, device, temp=0.05):
    """无监督的损失函数
    y_pred (tensor): bert的输出, [batch_size * 2, 768]

    """

    # 得到y_pred对应的label, [1, 0, 3, 2, ..., batch_size-1, batch_size-2]
    y_true = torch.arange(y_pred.shape[0], device=device)
    y_true = (y_true - y_true % 2 * 2) + 1
    # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
    # 将相似度矩阵对角线置为很小的值, 消除自身的影响
    sim = sim - torch.eye(y_pred.shape[0], device=device) * 1e12
    # 相似度矩阵除以温度系数
    sim = sim / temp
    # 计算相似度矩阵与y_true的交叉熵损失
    # 计算交叉熵，每个case都会计算与其他case的相似度得分，得到一个得分向量，目的是使得该得分向量中正样本的得分最高，负样本的得分最低
    loss = F.cross_entropy(sim, y_true)
    return torch.mean(loss)
#print(simcse_unsup_loss(pred, device))

def simcse_sup_loss(y_pred, device, lamda=0.05):
    """
    有监督损失函数
    """
    similarities = F.cosine_similarity(y_pred.unsqueeze(0), y_pred.unsqueeze(1), dim=2)
    row = torch.arange(0, y_pred.shape[0], 3)
    col = torch.arange(0, y_pred.shape[0])
    col = col[col % 3 != 0]

    similarities = similarities[row, :]
    similarities = similarities[:, col]
    similarities = similarities / lamda

    y_true = torch.arange(0, len(col), 2, device=device)
    loss = F.cross_entropy(similarities, y_true)
    return loss

# print(simcse_sup_loss(pred, device))

def contrastive_loss(x, x_aug, T= 0.05):
    """
    :param x: the hidden vectors of original data
    :param x_aug: the positive vector of the auged data
    :param T: temperature
    :return: loss
    """
    batch_size, _ = x.size()
    x_abs = x.norm(dim=1)
    x_aug_abs = x_aug.norm(dim=1)

    sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
    sim_matrix = torch.exp(sim_matrix / T)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = - torch.log(loss).mean()
    return loss

blank_states = torch.randn(4, 8)
groundtruth_outputs = torch.randn(4, 8)
print(blank_states)
print(groundtruth_outputs)
print(contrastive_loss(blank_states, blank_states))
# print(blank_states.shape[0])
# print(blank_states[0])
# print(groundtruth_outputs[0])
# sim_cl = [[0] * blank_states.shape[1] for i in range(2 * blank_states.shape[0])]  #创建一个二维值0列表，shape: 2 * blank_states.shape[0] *  blank_states.shape[1]
# for i in range(0,2*blank_states.shape[0],2):
#     # simcl[i] = torch.cat((blank_states[i], groundtruth_outputs[i]), 0)
#     sim_cl[i] = (blank_states[int(i/2)])
#     sim_cl[i+1] = (groundtruth_outputs[int(i/2)])
# # sim = [[blank_states[i],groundtruth_outputs[i]] for i in range(blank_states.shape[0])]
# print(sim_cl)
# # print(np.array(simcl).shape)
# # print(torch.cat((x, x, x), 1))
#
# dp = [[[0] * 4 for i in range(3)] for j in range(2)]  #创建2 * 3 * 4的三维列表
# print(np.array(dp).shape)
# print(dp)

a = [3,3,3,3,3,3,3]
print(a)

