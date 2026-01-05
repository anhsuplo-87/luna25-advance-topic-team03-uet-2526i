import torch
import torch.nn as nn
import torch.nn.functional as F

def cal_corr_conv(L):
    L = torch.flatten(L, start_dim=1)
    G = torch.matmul(L, L.T)
    G = torch.triu(G, diagonal=1)
    return torch.sqrt(torch.sum(G ** 2))


def corregularization(model):
    out = 0
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            out += cal_corr_conv(m.weight)

    return out


def rot_flip(x):    # N, C, X, Y, Z
    flipxyz = (torch.where(torch.rand(3) < 0.5)[0] + 2).tolist()    # 2, 3, 4 random extraction
    permxyz = [0, 1] + (torch.randperm(3) + 2).tolist()             # [0, 1, perm(2, 3, 4)]                        # 
    x = x.flip(dims=flipxyz)
    x = x.permute(permxyz)
    
    return x


def rot_flip_yz(x):    # N, C, X, Y, Z
    flipyz = (torch.where(torch.rand(2) < 0.5)[0] + 3).tolist()    # 3, 4 random extraction
    permyz = [0, 1, 2] + (torch.randperm(2) + 3).tolist()             # [0, 1, 2, perm(3, 4)]                        # 
    x = x.flip(dims=flipyz)
    x = x.permute(permyz)
    
    return x

def rot_flip_xy(x):    # N, C, X, Y, Z
    flipyz = (torch.where(torch.rand(3) < 0.5)[0] + 2).tolist()    # 2, 3 random extraction
    permyz = [0, 1] + (torch.randperm(2) + 2).tolist() + [4]             # [0, 1, perm(2, 3), 4]                        # 
    x = x.flip(dims=flipyz)
    x = x.permute(permyz)
    
    return x


def rot_flip_yz_2(x1, x2):    # N, C, X, Y, Z
    
    flipyz = (torch.where(torch.rand(2) < 0.5)[0] + 3).tolist()    # 3, 4 random extraction
    permyz = [0, 1, 2] + (torch.randperm(2) + 3).tolist()             # [0, 1, 2, perm(3, 4)]                        # 
    x1 = x1.flip(dims=flipyz)
    x2 = x2.flip(dims=flipyz)
    x1 = x1.permute(permyz)
    x2 = x2.permute(permyz)
    
    return x1, x2