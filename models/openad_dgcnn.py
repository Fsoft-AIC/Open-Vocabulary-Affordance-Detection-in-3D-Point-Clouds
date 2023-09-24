import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import clip


# function to find kNN indexes of every points in the cloud
def knn(x, k):      # shape of x: batch_size x fdim x 2048
    inner = -2*torch.matmul(x.transpose(2, 1), x)       # batch_size x 2048 x 2048
    xx = torch.sum(x**2, dim=1, keepdim=True)       # batch_size x 1 x 2048
    pairwise_distance = -xx - inner - xx.transpose(2, 1)        # batch_size x 2048 x 2048

    # (batch_size, num_points, k)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]        # batch_size x 2048 x k
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):       # what is dim9?
    batch_size = x.size(0)      # get the batch_size
    num_points = x.size(2)      # get number of points
    x = x.view(batch_size, -1, num_points)      # maybe not change the shape of x
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)       # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(
        0, batch_size, device=device).view(-1, 1, 1)*num_points     # 16 x 1 x 1 [[[0, 2048, 4096, 6144...

    idx = idx + idx_base

    idx = idx.view(-1)      # 1D vector

    _, num_dims, _ = x.size()

    # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature      # (batch_size, 2*num_dims, num_points, k)


class Transform_Net(nn.Module):
    def __init__(self, args):
        super(Transform_Net, self).__init__()
        self.args = args
        self.k = 3

        # BatchNorm layers
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        # Convolution layers
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        # Linear layers
        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 3*3)        # get a 3*3 vector
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        batch_size = x.size(0)

        # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv1(x)
        # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = self.conv2(x)
        # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)
        x = x.max(dim=-1, keepdim=False)[0]

        # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = self.conv3(x)
        # (batch_size, 1024, num_points) -> (batch_size, 1024)
        x = x.max(dim=-1, keepdim=False)[0]

        # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)
        # (batch_size, 512) -> (batch_size, 256)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)

        # (batch_size, 256) -> (batch_size, 3*3)
        x = self.transform(x)
        # (batch_size, 3*3) -> (batch_size, 3, 3)
        x = x.view(batch_size, 3, 3)

        return x

class ClassEncoder(nn.Module):
    def __init__(self):
        super(ClassEncoder, self).__init__()
        self.device = torch.device('cuda')
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        for param in self.clip_model.parameters():
            param.requires_grad = False
    def forward(self, classes):
        tokens = clip.tokenize(classes).to(self.device)
        text_features = self.clip_model.encode_text(tokens).to(self.device).permute(1, 0).float()
        return text_features

cls_encoder = ClassEncoder()

class OpenAD_DGCNN(nn.Module):
    def __init__(self, args, num_classes):
        super(OpenAD_DGCNN, self).__init__()
        self.args = args
        self.k = args.k
        self.transform_net = Transform_Net(args)
        self.num_classes = num_classes

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(1216, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv9 = nn.Conv1d(256, 512, 1)
        self.bn11 = nn.BatchNorm1d(512)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, x, affordance):
        batch_size = x.size(0)      # get the batch size
        num_points = x.size(2)      # get the number of points

        # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x0 = get_graph_feature(x, k=self.k)
        t = self.transform_net(x0)              # (batch_size, 3, 3)
        # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        x = x.transpose(2, 1)
        # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        x = torch.bmm(x, t)     # block matrix multiplication
        # (batch_size, num_points, 3) -> (batch_size, 3, num_points)
        x = x.transpose(2, 1)

        # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = get_graph_feature(x, k=self.k)
        # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv1(x)
        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)
        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x1 = x.max(dim=-1, keepdim=False)[0]

        # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = get_graph_feature(x1, k=self.k)
        # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv3(x)
        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)
        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x2 = x.max(dim=-1, keepdim=False)[0]

        # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = get_graph_feature(x2, k=self.k)
        # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv5(x)
        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x3 = x.max(dim=-1, keepdim=False)[0]

        # (batch_size, 64*3, num_points)
        x = torch.cat((x1, x2, x3), dim=1)

        # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = self.conv6(x)
        # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)
        x = x.max(dim=-1, keepdim=True)[0]

        # (batch_size, num_categoties, 1)
        # l = l.view(batch_size, -1, 1)
        # (batch_size, num_categoties, 1) -> (batch_size, 64, 1)
        # l = self.conv7(l)

        # x = torch.cat((x, l), dim=1)            # (batch_size, 1088, 1)
        # (batch_size, 1088, num_points)
        x = x.repeat(1, 1, num_points)

        # (batch_size, 1088+64*3, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)

        # (batch_size, 1088+64*3, num_points) -> (batch_size, 256, num_points)
        x = self.conv8(x)

        x = self.bn11(self.conv9(x)).permute(0, 2, 1).float()
        with torch.no_grad():
            text_features = cls_encoder(affordance)    # il [512, 19]
        x = (self.logit_scale * (x @ text_features) / (torch.norm(x, dim=2, keepdim=True) @ torch.norm(text_features, dim=0, keepdim=True))).permute(0, 2, 1)
        x = F.log_softmax(x, dim=1)
        return x