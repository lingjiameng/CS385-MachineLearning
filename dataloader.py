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



train_data, train_label = load_data_set(1,8)
print(train_data.shape)
print(train_label.shape)

train_set = np.concatenate([train_data,train_label],axis=1)

data_iter = torch.utils.data.DataLoader(train_set,shuffle=True,batch_size=64)
data_size = len(train_set)




# 随机抽样 和 batch

LR = 0.002
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
test_data, test_label = load_data_set(9,10)

X = test_data

Y = test_label
# print(Y)

_Y = mycls.predict(X)

error = np.sum(np.abs(Y-_Y)/2)
print("[correct rate:{:.5f}][test size:{}][error size:{}]".format(1 - error/len(Y),len(Y),error))

plt.plot(mycls.w)
plt.show()

