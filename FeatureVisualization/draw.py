import os
import argparse
from time import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import resnet

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import offsetbox
from sklearn import (manifold, decomposition, discriminant_analysis)
import cv2
import sklearn.cluster
import sklearn.metrics

import PIL

import numpy as np
# import pandas as pd

# import warnings

# warnings.filterwarnings('ignore')

SAVEDIR = "vis/"

def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation
    return modified_z_score > thresh


def not_outliers(points, thresh=3.5):
    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation
    return modified_z_score <= thresh


def count_label(y):
    """
    计算总的label个数
    """
    labels = []
    for label in y:
        if label not in labels:
            labels.append(label)
    return len(labels) 


def norm(arr):
    """
    标准化数据
    """
    _mu = np.mean(arr, axis=0)
    _var = np.var(arr, axis=0)
    _res = np.nan_to_num((arr - _mu)/_var)
    assert np.sum(_res == np.nan) == 0
    return _res

def tensor2np(t):
    """
    tensor(n,....) to np(n,-1)
    """
    return t.detach().numpy().reshape(t.shape[0], -1)


def plot_embedding_2D(X, y=None, title=None, savepath=None, sub="111",fig = None):
    """
    # Scale and visualize the embedding vectors

    # X is the projection location in 2D

    # y is index of labels #标签的下标

    """
    assert X.shape[1]==2,"error! require 2D points"
    assert len(X)==len(y),"error! require X has same length to y"

    n_label = count_label(y) 

    #归一化画布坐标为scale坐标即比例坐标
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) * 1.0 / (x_max - x_min)

    if fig is not None:
        ax = fig.add_subplot(sub)
    else:
        ax = plt.subplot(sub)

    #标出数据的分布
    for i in range(X.shape[0]):
        ax.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.jet(y[i] / n_label),
                 fontdict={'weight': 'bold', 'size': 7},
                 alpha=0.5)
    
    if title is not None:
        plt.title(title)
    if savepath is not None:
        plt.tight_layout()
        plt.savefig(savepath+"-2D.png", dpi=200)


def plot_embedding_3D(X, y=None, title=None,savepath=None,sub="111",fig=None):
    """
    # Scale and visualize the embedding vectors

    # X is the projection location in 3D

    # y is index of labels #标签的下标

    """
    assert X.shape[1] == 3, "error! require 3D points"
    assert len(X) == len(y), "error! require X has same length to y"

    n_label = count_label(y)

    if fig is not None:
        ax = fig.add_subplot(sub, projection='3d')
    else:
        ax = plt.subplot(sub, projection='3d')

    for i in range(X.shape[0]):
        ax.scatter(X[i, 0], X[i, 1], X[i, 2],
                   color=plt.cm.jet(y[i] / n_label),alpha = 0.5)

    if title is not None:
        plt.title(title)
    if savepath is not None:
        plt.tight_layout()
        plt.savefig(savepath+"-3D.png", dpi=200)
    

def cluster_score(X,y):
    """获取聚类结果
    
    Kmeans with ARI+AMI

    Adjusted Rand index score 调整兰德系数
    Adjusted Mutual Information score 调整互信息
    两者值越大相似度越高聚类效果越好
    https://blog.csdn.net/u010159842/article/details/78624135
    """

    n_label = count_label(y)
    myKmeans = sklearn.cluster.KMeans(n_clusters=n_label)
    x_cluster = myKmeans.fit_predict(X)

   
    ARI = sklearn.metrics.adjusted_rand_score(y, x_cluster)
    AMI = sklearn.metrics.adjusted_mutual_info_score(
        y, x_cluster, average_method='arithmetic')

    return ARI, AMI



def load_model(model,path):
    print("=> loading checkpoint '{}'".format(path))
    checkpoint = torch.load(path, map_location='cpu')
    best_prec1 = checkpoint['best_prec1']

    # fix bug that have additional "module."
    state_dict ={k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}

    model.load_state_dict(state_dict)
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(path, checkpoint['epoch']))


def foramt_train(path="npy/", start = -1):
    # 加载模型
    model = resnet.resnet20()
    load_model(model, "./save/save_0.1_resnet20/checkpoint.th")

    #加载数据集
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
   # 预训练LDA的数据集加载
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=1000, shuffle=False, pin_memory=False)

    for batch, (X_input, y) in enumerate(train_loader):
        if batch<start:
            continue
        print(">>>>>>saving for batch ",batch)
        LDA_Xs = model.forward_all(X_input)
        X_raw = tensor2np(X_input)
        y = y.detach().numpy()
        np.save(path+"{:0>2d}Y.npy".format(batch),y)
        #np.save(path+"{:0>2d}X0.npy".format(batch), X_raw)
        for layer, X in enumerate(LDA_Xs):
            X2 = tensor2np(X)
            name = "{:0>2d}X{}.npy".format(batch, layer)
            np.save(path+name, X2)
        del X_input,LDA_Xs,X_raw,X2,y





def load_layer_data(layer=0):
        # load data set from [start to end]
    data = []
    label = []
    for i in range(12):
        # print("loading data {:.2f} for layer {}".format((i+1)/33.0,layer))
        X_path = "npy/{:0>2d}X{}.npy".format(i,layer)
        Y_path = "npy/{:0>2d}Y.npy".format(i)
        data.append(np.load(X_path))
        label.append(np.load(Y_path))
    data = np.concatenate(data)
    label = np.concatenate(label)
    return data, label




def draw_LDA():
    # 超参数
    # hyper parameters
    

    # 加载模型
    model =resnet.resnet20()
    load_model(model,"./save/save_0.1_resnet20/checkpoint.th")
    
    #加载数据集
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=1000, shuffle=True)

    ########################################
    # 预训练LDA的数据集加载
    #LDA预训练
    LDAs = []
    accs = []
    for layer in range(7):
        X2,y = load_layer_data(layer)
        
        if(X2.shape[1]>10000):
            X2 = X2[:7000]
            y = y[:7000]
        LDA = discriminant_analysis.LinearDiscriminantAnalysis(n_components=3)
        print("data shape",X2.shape,"label",y.shape,count_label(y))
        LDA.fit(X2, y)
        score = LDA.score(X2,y)
        accs.append(score)
        print(">>>>>>>layer",layer,"train acc:",score)
        LDAs.append(LDA)
    ##############
    print("==========LDA pretrain finished!========")
    del X2
    del y
    ################################################3


    all_score = []
    # 提取部分数据集
    for batch, (X_input,y) in enumerate(val_loader):
        batch_score = []
        
        y = y.detach().numpy()
        
        #预测
        Xs = model.forward_all(X_input)

        # 分类正确率
        X_res = Xs[-1].detach().numpy()
        print("resnet20 acc:",np.mean(np.argmax(X_res,axis=1)==y))

        
        #分层画出结果
        for layer, X in enumerate(Xs):
        
            #print("===========layer:{}============".format(layer))
            X = tensor2np(X)
            #print(X.shape)
 
            # 我们有了每层的数据 X 和 Y
            vmodel = LDAs[layer]
            X_ = vmodel.transform(X)
            score = vmodel.score(X, y)
            
            batch_score.append(score)
            
            title = "LDA-{}".format(layer)
            print(title,"[Accuracy:{:.5f}]".format(score))

            if batch == 0: 
                y_ = y[not_outliers(X_)]
                X_draw = X_[not_outliers(X_)]
                
                fig = plt.figure()
                plot_embedding_3D(X_draw,y_,savepath=SAVEDIR+title, fig=fig,sub="111")
                fig = plt.figure()
                plot_embedding_2D(X_draw[:,:2], y_, savepath=SAVEDIR+title, fig=fig, sub="111")
            
        all_score.append(batch_score)
        if batch>=2:
            break
            
    print("==============final================")
    print("train acc:\n",np.array(accs))
    print("test acc:\n",np.array(all_score))
    print("test acc mean\n",np.mean(all_score,axis=0))


def draw_PCA():
    # 超参数
    # hyper parameters
    #SAVEDIR = "vis13/"

    # 加载模型
    model =resnet.resnet20()
    load_model(model,"./save/save_0.1_resnet20/checkpoint.th")
    
    #加载数据集
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=1000, shuffle=True)

    
    all_score = []
    # 提取部分数据集
    for batch, (X_input,y) in enumerate(val_loader):
        batch_score = []
        
        y = y.detach().numpy()
        
        #预测
        Xs = model.forward_all(X_input)

        # 分类正确率
        X_res = Xs[-1].detach().numpy()
        print("resnet20 acc:",np.mean(np.argmax(X_res,axis=1)==y))

        
        #分层画出结果
        for layer, X in enumerate(Xs): 
            #print("===========layer:{}============".format(layer))
            X = tensor2np(X)
            #print(X.shape)
 
            # 我们有了每层的数据 X 和 Y
            vmodel = decomposition.PCA(n_components=3)
            X_ = vmodel.fit_transform(X)
            ARI, AMI = cluster_score(X_, y)
            score = [ARI, AMI] + [i for i in vmodel.explained_variance_ratio_]
            batch_score.append(score)
            title = "PCA-{}".format(layer)
            print(title,"ARI, AMI , vars :{}".format(score))

            if batch == 0: 
                y_ = y[not_outliers(X_)]
                X_draw = X_[not_outliers(X_)]
                
                fig = plt.figure()
                plot_embedding_3D(X_draw,y_,savepath=SAVEDIR+title, fig=fig,sub="111")
                fig = plt.figure()
                plot_embedding_2D(X_draw[:,:2], y_, savepath=SAVEDIR+title, fig=fig, sub="111")
            
        all_score.append(batch_score)
        if batch>=2:
            break
    print("==============final================")
    print("ARI, AMI, var1,var2,var3 \n",np.array(all_score))
    print("ARI, AMI, var1,var2,var3 means\n",np.mean(all_score,axis=0))


def draw_TSNE():
    # 超参数
    # hyper parameters
    #SAVEDIR = "vis13/"

    # 加载模型
    model =resnet.resnet20()
    load_model(model,"./save/save_0.1_resnet20/checkpoint.th")
    
    #加载数据集
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=1000, shuffle=True)

    
    all_score = []
    # 提取部分数据集
    for batch, (X_input,y) in enumerate(val_loader):
        batch_score = []
        
        y = y.detach().numpy()
        
        #预测
        Xs = model.forward_all(X_input)

        # 分类正确率
        X_res = Xs[-1].detach().numpy()
        print("resnet20 acc:",np.mean(np.argmax(X_res,axis=1)==y))

        
        #分层画出结果
        for layer, X in enumerate(Xs): 
            #print("===========layer:{}============".format(layer))
            X = tensor2np(X)
            #print(X.shape)
 
            # 我们有了每层的数据 X 和 Y
            vmodel = manifold.TSNE(n_components=3)
            X_ = vmodel.fit_transform(X)
            ARI, AMI = cluster_score(X_, y)
            score = [ARI, AMI, vmodel.kl_divergence_]
            batch_score.append(score)
            title = "TSNE-{}".format(layer)
            print(title,"ARI, AMI , vars :{}".format(score))

            if batch == 0: 
                y_ = y[not_outliers(X_)]
                X_draw = X_[not_outliers(X_)]
                
                fig = plt.figure()
                plot_embedding_3D(X_draw,y_,savepath=SAVEDIR+title, fig=fig,sub="111")
                fig = plt.figure()
                plot_embedding_2D(X_draw[:,:2], y_, savepath=SAVEDIR+title, fig=fig, sub="111")
            
        all_score.append(batch_score)
        if batch>=2:
            break
    print("==============final================")
    print("ARI, AMI, kl \n",np.array(all_score))
    print("ARI, AMI, kl means\n",np.mean(all_score,axis=0))


parser = argparse.ArgumentParser(
    description='visual for resnet with PCA LDA and TSNE')

parser.add_argument("--model",default="" ,type=str,help="pca, lda or tsne")
parser.add_argument("--start",default=-1 ,type=int,help="the number you are killed last time")

if __name__ == "__main__":
    args = parser.parse_args()
    models = ["lda","pca","tsne","format"]
    assert args.model in models,"error! no such model!"
    with torch.no_grad():
        if args.model == "pca":
            draw_PCA()
        elif args.model == "lda":
            draw_LDA()
        elif args.model == "tsne":
            draw_TSNE()
        elif args.model == "format":
            foramt_train(start = args.start)
        
