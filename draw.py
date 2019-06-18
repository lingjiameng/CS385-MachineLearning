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



def normpdf(mu, cov, x):
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
    y = normpdf(mu, var, x)
    return x, y

def draw_logistic():
    # train_data, train_label = load_data_set(1,8)
    train_data, train_label = dl.load_train_norm()

    print(train_data.shape)
    print(train_label.shape)

    train_set = np.concatenate([train_data, train_label], axis=1)

    data_iter = torch.utils.data.DataLoader(
        train_set, shuffle=True, batch_size=64)
    data_size = len(train_set)

    # 随机抽样 和 batch
    # LR = 0.002
    LR = 0.002
    epoch = 50

    mycls = mymodel.LogisticClassifier(900, LR,loss="SGLD")

    # list of epoch loss and accuracy
    e_loss_list = []
    e_acc_list = []

    # list of batch accuray
    b_loss_list = []
    b_acc_list = []
    for e in range(epoch):
        e_loss = 0.0
        e_acc = 0.0

        for data in data_iter:
            # print(data.shape)
            # X = np.ndarray((64,900))
            # Y = np.ndarray((64,1))
            X = np.asarray(data[:, :900])
            Y = np.asarray(data[:, 900:])

            loss, acc = mycls.fit(X, Y)

            e_loss += loss
            e_acc += acc
            if 0 == e:
                b_loss_list.append(loss/len(Y))
                b_acc_list.append(acc/len(Y))

        if 0 == e:
            ## 画出一个epoch的loss和acc的变化曲线
            plt.subplot("121")
            plt.plot(b_loss_list)
            plt.xlabel('batches(batch_size=64)')
            plt.ylabel('average loss')
            plt.title("loss")

            plt.subplot("122")
            plt.plot(b_acc_list, color="coral")
            plt.xlabel('batches(batch_size=64)')
            plt.ylabel('average accuracy')
            plt.title("accuracy")

            plt.suptitle("loss and accuracy of first epoch")
            plt.show()

        e_loss /= data_size
        e_acc /= data_size
        e_loss_list.append(e_loss)
        e_acc_list.append(e_acc)
        print("[epoch:{}][loss:{:.5f}][acc:{:.5f}]".format(e, e_loss, e_acc))

    ## 画出整个training的loss和acc的变化曲线
    plt.subplot("121")
    plt.plot(e_loss_list)
    plt.xlabel('epochs')
    plt.ylabel('average loss')
    plt.title("loss of 50 epochs")

    plt.subplot("122")
    plt.plot(e_acc_list, color="coral")
    plt.xlabel('epochs')
    plt.ylabel('average accuracy')
    plt.title("accuracy of 50 epochs")
    plt.show()

    # test data
    X, Y = dl.load_test_norm()

    _Y = mycls.predict(X)

    error = np.sum(np.abs(Y-_Y)/2)
    print("[correct rate:{:.5f}][test size:{}][error size:{}]".format(
        1 - error/len(Y), len(Y), error))

    plt.subplot("121")
    plt.plot(mycls.w)
    plt.ylabel("value of weight")
    plt.xlabel(r"dimension of $\beta$")
    plt.subplot("122")
    plt.xlabel("value of weight")
    plt.ylabel("num of such weight")
    plt.hist(mycls.w, bins=200)
    plt.show()

def draw_lda():
    ## get train data
    train_data, train_label = dl.load_train_norm()
    data_pos = train_data[np.squeeze(train_label == 1), :]
    data_neg = train_data[np.squeeze(train_label == -1), :]
    print("[train data{}]".format(train_data.shape))

    # get mean and covariance
    mu_pos = data_pos.mean(axis=0).reshape(-1, 1)
    mu_neg = data_neg.mean(axis=0).reshape(-1, 1)
    cov_pos = np.cov(data_pos, rowvar=False)
    cov_neg = np.cov(data_neg, rowvar=False)

    # get s_B and s_W
    delta_mu = mu_pos-mu_neg
    SB = np.matmul(delta_mu, delta_mu.T)
    SW = (len(data_pos)*cov_pos + len(data_neg)*cov_neg)
    SW_inv = np.linalg.inv(SW)
    # tmp = SW_inv - SW_inv.T
    # print(len(tmp[tmp <= 1e-15]))
    swsb = np.matmul(SW_inv, SB)
    # TODO: swsb 不对称

    landa, beta = np.linalg.eig(swsb)
    #TODO: eigh 只适用于对称矩阵，eig适用于所有方阵

    print("eigenvalues number: ", len(landa),
          " eigenvalues(module<=1e-15): ", len(landa[np.abs(landa) <= 1e-15]))
    print("first eigenvector(module<=1e-6): ",
          len(beta[:, 0][np.abs(beta[:, 0]) <= 1e-6]))
    # print(landa[0], beta[:,0])

    # print(landa)
    #画特征值的模
    plt.yscale("log")
    plt.grid()
    plt.scatter(np.arange(900), np.abs(landa), s=8)
    plt.xlabel('eigenvalues index', fontsize=16)
    plt.ylabel('module of eigenvalue ( log scale )', fontsize=16)
    # plt.title("eigenvalue modules", fontsize=16)
    plt.show()

    #特征值的复平面图像
    # print (landa.imag)

    # 所有特征值在复平面上的展示
    plt.yscale('symlog', linthreshy=1e-22)
    plt.xscale("symlog", linthreshx=1e-22)
    plt.scatter(landa.real, landa.imag, s=6)
    plt.ylabel(
        "imaginary part of eigenvalues ( linear scale while $|Y| <= 10^{-22}$ and log scale while $|Y| > 10^{-22}$ )")
    plt.xlabel(
        "real part of eigenvalues ( linear scale while $|X| <= 10^{-22}$ and log scale while  $|X| > 10^{-22}$ )", fontsize=16)
    plt.title("eigenvalues in complex plane", fontsize=16)
    plt.grid()
    plt.show()

    #分类效果最好的向量为
    beta_star = beta[:, 0].real

    #推倒出的beta 归一化处理
    beta_ = np.matmul(np.linalg.inv(SW), delta_mu)
    beta_ = beta_[:, 0]
    beta_ = np.divide(beta_, np.sqrt(np.dot(beta_, beta_)))

    # lan = np.matmul(swsb, beta_)/beta_
    # print("lan: ",np.mean(lan),np.max(lan),np.min(lan),lan.shape )
    # lan2 = beta[:, 0].real/beta_
    # print("lan2: ", np.mean(lan2), np.max(lan2), np.min(lan2), lan.shape)

    #计算出的特征值与推倒结论分类效果对比
    z = np.matmul(train_data, beta_star)[:, np.newaxis]
    z_ = np.matmul(train_data, beta_)[:, np.newaxis]
    print(z.shape)
    ##求出的特征向量投影效果
    plt.subplot("121")
    plt.hist(z[train_label == 1], bins=200, alpha=0.5, label="positive data")
    plt.hist(z[train_label == -1], bins=200, alpha=0.5, label="negative data")
    plt.title("eigenvector with max eigenvalue", fontsize=16)
    plt.legend(fontsize=16)

    ##推论的特征向量投影效果
    plt.subplot("122")
    plt.hist(z_[train_label == 1], bins=200, alpha=0.5, label="positive data")
    plt.hist(z_[train_label == -1], bins=200, alpha=0.5, label="negative data")
    plt.title(
        "eigenvector $S_W^{-1}(\mu^+ - \mu^-)$ (normalized)", fontsize=16)
    plt.legend(fontsize=16)
    plt.show()

    #对比两个特征向量的分布
    plt.plot(beta_star, alpha=0.8)
    plt.plot(beta_, alpha=0.8, color="coral")
    plt.title("comparison of two vectors", fontsize=16)
    plt.show()

    #展示了 (特征值较小小的实数特征向量) 的投影结果
    indexs = np.argsort(np.abs(landa))
    count = 0
    for i in range(1, 900, 1):
        eig = beta[:, indexs[i]].reshape(-1, 1)
        if np.sum(eig.imag > 0) > 0:
            continue
        count += 1
        # print(eig.shape)
        eig = eig.real
        z = np.matmul(train_data, eig)
        # print(z)
        plt.subplot("23{}".format(count))
        plt.hist(z[train_label == 1], bins=100,
                 alpha=0.5, label="positive data")
        plt.hist(z[train_label == -1], bins=100,
                 alpha=0.5, label="negative data")
        plt.title("{}-th eigenvector with value {:.2e}".format(i,
                                                               landa[indexs[i]].real))
        plt.legend()
        if count >= 6:
            plt.suptitle("Projection Result")
            plt.show()
            break
    # print(count)


def draw_svm():
    # train_data, train_label = load_data_set(1,8)
    train_data, train_label = dl.load_train_norm()
    print("[train data shape: {}]".format(train_data.shape))

    ####### 节省时间，直接load训练结果
    # mysvm = sklearn.svm.SVC(verbose=True, gamma="auto")
    # mysvm.fit(train_data, np.squeeze(train_label))
    mysvm = mymodel.load_model("save/svm.model")

    # test data
    test_data, test_label = dl.load_test_norm()

    predict_label = mysvm.predict(test_data)

    error = np.sum(np.squeeze(test_label) != predict_label)

    print("=====================testing result=====================")
    print("[correct rate:{:.5f}][test size:{}][error size:{}]".format(
        1 - error/len(test_label), len(test_label), error))

    support_vector = mysvm.support_vectors_
    sv_label = train_label[mysvm.support_, 0]

    ################ 用LDA展示支持向量
    print(len(mysvm.support_))
    mylda = mymodel.LDA()
    mylda.fit(train_data, train_label)
   

    z = np.matmul(train_data, mylda.w)
    sv = np.matmul(support_vector, mylda.w)
    

    plt.subplot("122")
    plt.hist(z[train_label == 1], bins=200, alpha=0.35, label="positive data")
    plt.hist(sv[sv_label == 1], bins=200,
             alpha=0.35, label="support vector(+1)")
    plt.legend()
    # plt.show()

    plt.subplot("121")
    plt.hist(z[train_label == -1], bins=200, alpha=0.35, label="negative data")

    plt.hist(sv[sv_label == -1], bins=200,
             alpha=0.35, label="support vector(-1)")
    plt.legend()
    plt.show()

    ###########################################3
    plt.hist(z[train_label == 1], bins=200, alpha=0.35, label="positive data")
    plt.hist(z[train_label == -1], bins=200, alpha=0.35, label="negative data")
    plt.hist(sv, bins=200,
             alpha=0.35, label="support vector")
    plt.legend(fontsize = 20)
    plt.show()
    #########################

    # ###用PCA降维展示数据
    # tofix 效果很差
    # print(support_vector.shape)
    # myLinearPCA = mymodel.LinearPCA(n_components=2)

    # myLinearPCA.fit(train_data)
    # x_proj = myLinearPCA.transform(train_data)
    # sv = myLinearPCA.transform(support_vector)

    # #投影后的结果聚类显示
    # x_label = np.squeeze(train_label)

    # plt.figure(1)
    # plt.scatter(x_proj[x_label == 1, 0],
    #             x_proj[x_label == 1, 1], alpha=0.4, s=4)
    # plt.scatter(x_proj[x_label == -1, 0],
    #             x_proj[x_label == -1, 1], alpha=0.4, s=4)
    # plt.scatter(sv[:,0],sv[:,1], alpha=1, s=4)
    # plt.show()



def draw_cnn():
    EPOCH = 50
    BATCH_SIZE = 64
    LR = 0.001              # learning rate

    train_set = dl.FaceVisionSet(train=True)
    test_set = dl.FaceVisionSet(train=False)
    print(train_set)
    print(test_set)

    train_loader = torch.utils.data.DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set,batch_size=BATCH_SIZE,shuffle=True)

    mycnn = mymodel.CNN()
    print(mycnn)  
    opt = torch.optim.Adam(params=mycnn.parameters(),lr=LR)
    loss_func = nn.CrossEntropyLoss()

    e_loss = 0.0
    e_acc = 0.0
    # list of epoch loss and accuracy
    e_loss_list = []
    e_acc_list = []

    # list of batch accuray
    b_loss_list = []
    b_acc_list = []
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

            if 0 == epoch:
                b_loss_list.append(loss/len(pred_y))
                b_acc_list.append(acc/len(pred_y))

        if 0 == epoch:
            ## 画出一个epoch的loss和acc的变化曲线
            plt.subplot("121")
            plt.plot(b_loss_list)
            plt.xlabel('batches(batch_size=64)')
            plt.ylabel('average loss')
            plt.title("loss")

            plt.subplot("122")
            plt.plot(b_acc_list, color="coral")
            plt.xlabel('batches(batch_size=64)')
            plt.ylabel('average accuracy')
            plt.title("accuracy")

            plt.suptitle("loss and accuracy of first epoch")
            plt.show()

        print(flush=True)
        e_loss = e_loss/ len(train_set)
        e_acc = e_acc / len(train_set)
        e_loss_list.append(e_loss)
        e_acc_list.append(e_acc)
        e_cpu_time = time.process_time() - cpu_time
        e_wall_time = time.time() - wall_time
        print("[epoch:{}][cpu time:{:.3f}][wall time:{:.3f}][loss:{:.5f}][acc:{:.5f}]".format(
            epoch, e_cpu_time, e_wall_time, e_loss, e_acc))
    
    ## 画出整个training的loss和acc的变化曲线
    plt.subplot("121")
    plt.plot(e_loss_list)
    plt.xlabel('epochs')
    plt.ylabel('average loss')
    plt.title("loss of 50 epochs")

    plt.subplot("122")
    plt.plot(e_acc_list, color="coral")
    plt.xlabel('epochs')
    plt.ylabel('average accuracy')
    plt.title("accuracy of 50 epochs")
    plt.show()



    #################################################
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




def analysis_pca():
    def norm(arr):
        mu = np.mean(arr, axis=0)
        var = np.var(arr, axis=0)
        print("norm", end=":")
        printshape([arr, mu, var])
        return (arr-mu)/var

    def printshape(l):
        for arr in l:
            print("|shape{}|".format(arr.shape), end="")
        print()
    # load data
    train_data, train_label = dl.load_train_norm()
    test_data, test_label = dl.load_test_norm()

    print(train_data.shape, train_label.shape)

    x = norm(train_data)

    #hyper parameters
    n_feature = 900
    n_components = 10

    # target weights
    w = np.zeros((n_feature, n_components))

    ###渐进式求解 与 直接求解的验证

    ## 直接求解
    xk = np.copy(x)
    for comp in range(n_component):
        xTx = np.matmul(xk.T, xk)
        eigva, eigve = np.linalg.eig(xTx)
        if not comp:
            x_score = np.matmul(x, eigve)
            print(np.var(x_score, axis=0)[:10])
        # break
        w_comp = eigve[np.argmax(np.abs(eigva))]  # 选取最优解

        w[:, comp] = w_comp  # 保存

        w_comp = w_comp.reshape(-1, 1)
        xw = np.matmul(x, w_comp)
        print("score:", np.var(xw))
        xk = xk - np.matmul(xw, w_comp.T)
        # print(xk==x)
        printshape([xTx, w_comp, xw, xk])

    x_proj = np.matmul(x, w)
    printshape([x_proj, train_label])
    x_label = np.squeeze(train_label)

    x_proj_var = np.var(x_proj, axis=0)

    comp_list = [np.argmin(x_proj_var)]

    for comp in comp_list:  # range(n_component):
        plt.scatter(x_proj[x_label == 1, 0],
                    x_proj[x_label == 1, comp], alpha=0.4, s=4)
        plt.scatter(x_proj[x_label == -1, 0]+50,
                    x_proj[x_label == -1, comp], alpha=0.4, s=4)
        plt.show()
    print("===========")
    print(w)


def draw_pca(hog="hog"):
    """
    hog = {"hog" , "img"}
    """
    # load data
    if hog == "hog":
        train_data, train_label = dl.load_test_norm()
    elif hog == "img":
        # load data
        train_data, train_label = dl.load_test_norm_img()
        train_data = train_data.reshape(len(train_data), -1)

    print(train_data.shape, train_label.shape)

    ##############
    ## PCA
    myLinearPCA = mymodel.LinearPCA(n_components=2)

    x_proj = myLinearPCA.fit_transform(train_data)

    #投影后的结果聚类显示
    x_label = np.squeeze(train_label)

    plt.figure(1)
    plt.scatter(x_proj[x_label == 1, 0],
                x_proj[x_label == 1, 1], alpha=0.4, s=4, label="positive data")
    plt.scatter(x_proj[x_label == -1, 0],
                x_proj[x_label == -1, 1], alpha=0.4, s=4, label="negative data")
    plt.title("PCA projection of face "+hog+" set")
    plt.legend()
    plt.show()


def draw_tsne(hog="hog"):
    """
    hog = {"hog" , "img"}
    """
    # load data
    if hog == "hog":
        train_data, train_label = dl.load_test_norm()
    elif hog == "img":
        # load data
        train_data, train_label = dl.load_test_norm_img()
        train_data = train_data.reshape(len(train_data), -1)
    print(train_data.shape, train_label.shape)

    ##############
    ## PCA
    # X_reduce = mymodel.LinearPCA(n_components=50).fit_transform(train_data)
    X_reduce = train_data

    ## t-SNE
    import sklearn.manifold
    X_embedded = sklearn.manifold.TSNE(
        n_components=2, init="pca").fit_transform(X_reduce)

    #投影后的结果聚类显示
    x_proj = X_embedded
    x_label = np.squeeze(train_label)

    plt.figure(1)
    plt.scatter(x_proj[x_label == 1, 0],
                x_proj[x_label == 1, 1], alpha=0.4, s=4, label="positive data")
    plt.scatter(x_proj[x_label == -1, 0],
                x_proj[x_label == -1, 1], alpha=0.4, s=4, label="negative data")
    plt.title("t-SNE projection of face "+hog+" set")
    plt.legend()
    plt.show()


parser = argparse.ArgumentParser(
    description='draw model with logistic lda svm cnn pca tsne')

parser.add_argument("--model", default="", type=str,
                    help="log lda svm and cnn pca tsne")
parser.add_argument("--data", default="hog", type=str,
                    help="hog img")

if __name__ == "__main__":
    args = parser.parse_args()
    models = ["lda", "log", "svm", "cnn","pca","tsne"]
    assert args.model in models, "error! no such model!"

    if args.model == "log":
        draw_logistic()
    elif args.model == "lda":
        draw_lda()
    elif args.model == "svm":
        draw_svm()
    elif args.model == "cnn":
        train_cnn()
    elif args.model == "pca":
        draw_pca(hog=args.data)
    elif args.model == "tsne":
        draw_tsne(hog=args.data)
