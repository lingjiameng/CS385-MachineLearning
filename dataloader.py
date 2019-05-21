import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data


import mymodel

train_data = []
train_label = []

def load_data_set(start,end):
    # load data set from [start to end]
    data = []
    label = []
    for i in range(start,end+1,1):
        hogs_path = "data/face_hogs{:0>2d}.npy".format(i)
        labels_path = "data/face_labels{:0>2d}.npy".format(i)
        data.append(np.load(hogs_path))
        label.append(np.load(labels_path))
    data = np.concatenate(data)
    label = np.concatenate(label)[:, np.newaxis]
    return data,label


def load_train_norm():
    train_data1, train_label1 = load_data_set(1, 4)
    train_data2, train_label2 = load_data_set(5, 8)
    
    train_data2_pos = train_data2[np.squeeze(train_label2==1),:]
    train_label2_pos = train_label2[train_label2==1][:,np.newaxis]

    data = np.concatenate([train_data1,train_data2_pos])
    label =np.concatenate([train_label1,train_label2_pos])
    return data,label

def load_test_norm():
    train_data1, train_label1 = load_data_set(10, 10)
    train_data2, train_label2 = load_data_set(9, 9)

    train_data2_pos = train_data2[np.squeeze(train_label2 == 1), :]
    train_label2_pos = train_label2[train_label2 == 1][:, np.newaxis]

    data = np.concatenate([train_data1, train_data2_pos])
    label = np.concatenate([train_label1, train_label2_pos])
    return data, label

def train_logistic():
    # train_data, train_label = load_data_set(1,8)
    train_data, train_label = load_train_norm()

    print(train_data.shape)
    print(train_label.shape)

    train_set = np.concatenate([train_data,train_label],axis=1)

    data_iter = torch.utils.data.DataLoader(train_set,shuffle=True,batch_size=64)
    data_size = len(train_set)

    # 随机抽样 和 batch
    # LR = 0.002
    LR= 0.002
    epoch = 50


    mycls  = mymodel.LogisticClassifier(900,LR)

    for e in  range(epoch):
        e_loss = 0.0
        e_acc = 0.0
        for data in data_iter:
            # print(data.shape)
            # X = np.ndarray((64,900))
            # Y = np.ndarray((64,1))
            X = np.asarray(data[:,:900])
            Y = np.asarray(data[:,900:])

            loss,acc = mycls.learn(X,Y)

            e_loss += loss
            e_acc += acc

        e_loss /= data_size
        e_acc /= data_size
        print("[epoch:{}][loss:{:.5f}][acc:{:.5f}]".format(e,e_loss,e_acc))


    # test data
    # test_data, test_label = load_data_set(9,10)
    test_data, test_label = load_test_norm()

    X = test_data

    Y = test_label
    # print(Y)

    _Y = mycls.predict(X)

    error = np.sum(np.abs(Y-_Y)/2)
    print("[correct rate:{:.5f}][test size:{}][error size:{}]".format(1 - error/len(Y),len(Y),error))

    plt.plot(mycls.w)
    plt.show()

def train_lda():
    train_data, train_label = load_train_norm()
    print(train_data.shape)
    print(train_label.shape)

    data_pos = train_data[np.squeeze(train_label==1),:]
    data_neg = train_data[np.squeeze(train_label == -1), :]

    # print(data_pos.shape)
    # print(data_neg.shape)

    mu_pos = data_pos.mean(axis=0).reshape(-1,1)
    mu_neg = data_neg.mean(axis=0).reshape(-1, 1)

    delta_mu = mu_pos-mu_neg
    # print(mu_pos.shape)
    SB = np.matmul(delta_mu , delta_mu.T)

    cov_pos = np.cov(data_pos.T)
    cov_neg = np.cov(data_neg.T)
    # print(cov_pos.shape)
    SW = (len(data_pos)*cov_pos + len(data_neg)*cov_neg)
    # SW = SW/len(train_data)

    print(np.linalg.slogdet(SW))

    landa,beta = np.linalg.eigh(np.matmul(np.linalg.inv(SW),SB),"U")

    plt.plot(landa)
    plt.show()
    print(np.dot(beta[0], beta[1]), np.dot(beta[1], beta[2]))
    # exit()

    
    
    print(landa.max(),landa.min())
    
    print(beta.shape)
    index = (np.abs(landa)).argmin()
    indexs = np.argsort(np.abs(landa))

    eig2 = np.matmul(np.linalg.inv(SW), delta_mu)

    z = np.matmul(train_data, eig2)
    # print(z.shape)
    plt.hist(z[train_label == 1], bins=200, alpha=0.5, label="pos")
    plt.hist(z[train_label == -1], bins=200, alpha=0.5, label="neg")
    plt.legend()

    plt.show()

    for i in range(-1,-10,-1):
        eig = beta[:, indexs[i]].reshape(-1, 1)
        # print(eig.shape)

        eig2 = np.matmul(np.linalg.inv(SW),delta_mu)

        z = np.matmul(train_data,eig)
        # print(z.shape)
        plt.hist(z[train_label==1],bins=200,alpha=0.5,label="pos")
        plt.hist(z[train_label == -1], bins=200, alpha=0.5, label="neg")
        plt.legend()

        plt.show()


if __name__ == "__main__":
    # train_logistic()
    train_lda()
