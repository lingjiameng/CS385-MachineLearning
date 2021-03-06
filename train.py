import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import matplotlib.pyplot as plt
import sklearn.svm
from sklearn.metrics import classification_report

import dataloader as dl
import mymodel


def normpdf(mu,cov,x):
    ##只适用于一维变量的分布
    ## x 可以为向量
    scalar = 1.0/np.sqrt(2*np.pi*cov)
    exp = np.exp(-1.0/(2*cov)*(x-mu)**2)
    return scalar*exp

def norm_curve(z):
    #给定数据集，返回近似正太分布曲线
    mu = np.mean(z)
    var = np.var(z)
    mvar = np.sqrt(var)
    x = np.arange(mu - 6*mvar, mu + 6*mvar, mvar/1000.0)
    y = normpdf(mu,var,x)
    return x,y

def train_logistic(loss = "SGD"):
    # train_data, train_label = load_data_set(1,8)
    train_data, train_label = dl.load_train_norm()
    print("[train data shape: {}]".format(train_data.shape))

    train_set = np.concatenate([train_data, train_label], axis=1)
    data_iter = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=64)
    data_size = len(train_set)

    LR = 0.002
    epoch = 50
    mycls = mymodel.LogisticClassifier(900, LR,loss=loss)

    print("=====================training result=====================")
    for e in range(epoch):
        e_loss = 0.0
        e_acc = 0.0
        for data in data_iter:
            X = np.asarray(data[:, :900])  # X = np.ndarray((64,900))
            Y = np.asarray(data[:, 900:])  # Y = np.ndarray((64,1))

            loss, acc = mycls.fit(X, Y)

            e_loss += loss
            e_acc += acc

        e_loss /= data_size
        e_acc /= data_size
    
        print("[epoch:{}][loss:{:.5f}][acc:{:.5f}]".format(e, e_loss, e_acc))


    # test data
    test_data, test_label = dl.load_test_norm()

    predict_label = mycls.predict(test_data)

    error = np.sum(test_label!=predict_label)
    
    print("=====================testing result=====================")
    print("[correct rate:{:.5f}][test size:{}][error size:{}]".format(
        1 - error/len(test_label), len(test_label), error))
    
    print("=====================saving model=====================")
    mymodel.save_model(mycls,"save/log.model")
    print("saving model to save/log.model finished!")

def train_lda():
    ## get train data
    train_data, train_label = dl.load_train_norm()
    test_data, test_label = dl.load_test_norm()
    mylda = mymodel.LDA(vis=True)

    mylda.fit(train_data,train_label)

    z = mylda.predict(test_data)

    error = np.sum(np.abs(z - test_label))/2

    print("[correct rate : {:.5f}][total size: {}][error size: {}]".format(
        1.0-error/len(z), len(z), error))
    
    print("=====================saving model=====================")
    mymodel.save_model(mylda, "save/lda.model")
    print("saving model to save/lda.model finished!")

def train_svm(kernel = "rbf"):
    # train_data, train_label = load_data_set(1,8)
    train_data, train_label = dl.load_train_norm()
    print("[train data shape: {}]".format(train_data.shape))

    mysvm = sklearn.svm.SVC(verbose=True,gamma="auto",kernel=kernel)
    mysvm.fit(train_data,np.squeeze(train_label))
    
    print()
    ######################
    # train data
    predict_label = mysvm.predict(train_data)

    error = np.sum(np.squeeze(train_label) != predict_label)
    print("=====================testing result=====================")
    print("[correct rate:{:.5f}][train size:{}][error size:{}]".format(
        1 - error/len(train_label), len(train_label), error))
    
    #############################
    # test data
    test_data, test_label = dl.load_test_norm()

    predict_label = mysvm.predict(test_data)

    error = np.sum(np.squeeze(test_label) != predict_label)

    print("=====================testing result=====================")
    print("[correct rate:{:.5f}][test size:{}][error size:{}]".format(
        1 - error/len(test_label), len(test_label), error))
    print("=====================saving model=====================")
    mymodel.save_model(mysvm, "save/svm.model")
    print("saving model to save/svm.model finished!")



def train_cnn():
    EPOCH = 10               
    BATCH_SIZE = 64
    LR = 0.001              # learning rate

    train_set = dl.FaceVisionSet(train=True)
    test_set = dl.FaceVisionSet(train=False)

    train_loader = torch.utils.data.DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set,batch_size=BATCH_SIZE,shuffle=True)

    mycnn = mymodel.CNN()
    print(mycnn)  
    opt = torch.optim.Adam(params=mycnn.parameters(),lr=LR)
    loss_func = nn.CrossEntropyLoss()

    e_loss = 0.0
    e_acc = 0.0

    for epoch in range(EPOCH):
        cpu_time = time.process_time()
        wall_time = time.time()
        for step,(b_x,b_y) in enumerate(train_loader):
            output = mycnn(b_x)[0]
            loss = loss_func(output,b_y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            e_loss += loss.data.numpy()
            pred_y = torch.max(output, 1)[1].data.numpy()

            acc = (pred_y == b_y.data.numpy()).astype(int).sum()
            e_acc += acc
            if step%50 == 0:
                print("*",end="",flush=True)
        print(flush=True)
        e_loss = e_loss/ len(train_set)
        e_acc = e_acc / len(train_set)
        e_cpu_time = time.process_time() - cpu_time
        e_wall_time = time.time() - wall_time
        print("[epoch:{}][cpu time:{:.3f}][wall time:{:.3f}][loss:{:.5f}][acc:{:.5f}]".format(
            epoch, e_cpu_time, e_wall_time, e_loss, e_acc))


    # test 

    cpu_time = time.process_time()
    wall_time = time.time()
    e_loss = 0.0
    e_acc = 0.0
    for step, (b_x, b_y) in enumerate(test_loader):
        output = mycnn(b_x)[0]
        loss = loss_func(output, b_y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        e_loss += loss.data.numpy()
        pred_y = torch.max(output, 1)[1].data.numpy()

        acc = (pred_y == b_y.data.numpy()).astype(int).sum()
        e_acc += acc


    e_loss = e_loss/ len(test_set)
    e_acc = e_acc / len(test_set)
    e_cpu_time = time.process_time() - cpu_time
    e_wall_time = time.time() - wall_time
    print("\n===========test result:=============")
    print("[cpu time:{:.3f}][wall time:{:.3f}][loss:{:.5f}][acc:{:.5f}]".format(
        e_cpu_time, e_wall_time, e_loss, e_acc))


    print("=====================saving model=====================")
    torch.save(mycnn, 'save/cnn.pkl')
    print("saving model to save/cnn.pkl finished!")



parser = argparse.ArgumentParser(
    description='train model with logistic lda svm and cnn')

parser.add_argument("--model", default="", type=str,
                    help="log lda svm and cnn")
parser.add_argument("--loss", default="SGD", type=str,
                    help="SGD, SGLD")
parser.add_argument("--kernel", default="rbf", type=str,
                    help="linear, poly, rbf, sigmoid")
                   

if __name__ == "__main__":
    args = parser.parse_args()
    models = ["lda", "log", "svm", "cnn"]
    assert args.model in models, "error! empty model!"

    if args.model == "log":
        train_logistic(args.loss)
    elif args.model == "lda":
        train_lda()
    elif args.model == "svm":
        train_svm(args.kernel)
    elif args.model == "cnn":
        train_cnn()



