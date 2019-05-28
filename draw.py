import numpy as np
import torch.utils.data
import matplotlib.pyplot as plt
import sklearn.svm

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

    mycls = mymodel.LogisticClassifier(900, LR)

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

            loss, acc = mycls.learn(X, Y)

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

    mysvm = sklearn.svm.SVC(verbose=True, gamma="auto")
    mysvm.fit(train_data, np.squeeze(train_label))

    # test data
    test_data, test_label = dl.load_test_norm()

    predict_label = mysvm.predict(test_data)

    error = np.sum(np.squeeze(test_label) != predict_label)

    print("=====================testing result=====================")
    print("[correct rate:{:.5f}][test size:{}][error size:{}]".format(
        1 - error/len(test_label), len(test_label), error))

    ############# 用LDA展示支持向量
    print(len(mysvm.support_))
    mylda = mymodel.LDA()
    mylda.learn(train_data, train_label)
    support_vector = mysvm.support_vectors_

    z = np.matmul(train_data, mylda.w)
    sv = np.matmul(support_vector, mylda.w)
    sv_label = train_label[mysvm.support_, 0]

    plt.hist(z[train_label == 1], bins=200, alpha=0.35, label="positive data")
    plt.hist(sv[sv_label == 1], bins=200,
             alpha=0.35, label="support vector(+1)")
    plt.legend()
    plt.show()

    plt.hist(z[train_label == -1], bins=200, alpha=0.35, label="negative data")

    plt.hist(sv[sv_label == -1], bins=200,
             alpha=0.35, label="support vector(-1)")
    plt.legend()
    plt.show()
    #########################


if __name__ == "__main__":
    # draw_logistic()
    # print("+"*54)
    # draw_lda()
    draw_svm()
