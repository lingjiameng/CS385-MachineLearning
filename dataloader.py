import os

import numpy as np

np_hog_set_dir = "data/hog/"
np_img_set_dir = "data/img/"

def load_data_set(start,end):
    # load data set from [start to end]
    data = []
    label = []
    for i in range(start,end+1,1):
        hogs_path = np_hog_set_dir + "face_hogs{:0>2d}.npy".format(i)
        labels_path = np_hog_set_dir+ "face_labels{:0>2d}.npy".format(i)
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


def load_img_set(start,end):
     # load data set from [start to end]
    data = []
    label = []
    for i in range(start, end+1, 1):
        hogs_path = np_img_set_dir + "face_imgs{:0>2d}.npy".format(i)
        labels_path = np_img_set_dir + "face_labels{:0>2d}.npy".format(i)
        data.append(np.load(hogs_path))
        label.append(np.load(labels_path))
    data = np.concatenate(data)
    label = np.concatenate(label)[:, np.newaxis]
    return data, label


def load_train_norm_img():
    """
    加载标准的灰度图片数据集
    """
    train_data1, train_label1 = load_img_set(1, 4)
    train_data2, train_label2 = load_img_set(5, 8)

    train_data2_pos = train_data2[np.squeeze(train_label2 == 1), :]
    train_label2_pos = train_label2[train_label2 == 1][:, np.newaxis]

    data = np.concatenate([train_data1, train_data2_pos])
    label = np.concatenate([train_label1, train_label2_pos])
    return data, label


def load_test_norm_img():
    """
    加载标准的灰度图片数据集标签
    """
    train_data1, train_label1 = load_img_set(10, 10)
    train_data2, train_label2 = load_img_set(9, 9)

    train_data2_pos = train_data2[np.squeeze(train_label2 == 1), :]
    train_label2_pos = train_label2[train_label2 == 1][:, np.newaxis]

    data = np.concatenate([train_data1, train_data2_pos])
    label = np.concatenate([train_label1, train_label2_pos])
    return data, label


class FaceVisionSet(object):
    """
    train = True, False 则加载测试集
    """
    def __init__(self,train = True):
        self.train = train
        root = np_img_set_dir
        assert os.path.exists(root),"error: no such dir "+root
        
        self.data , self.labels = self._load()
    
    def _load(self):
        """
        加载数据集 
        data:(n,1,p,...,p)
        labels:(n,) {0,1}对应-1,1
        """
        data = None
        labels = None
        if self.train:
            data, labels = load_train_norm_img()
        else:
            data, labels = load_test_norm_img()
        
        #将数据转换为float32，并将扩充维数为 n*1*96*96 , 1 为通道数
        data = np.expand_dims(data.astype(np.float32),axis=1)
        # 数据归一化 
        data = data/255.0

        #将维数压缩为(n,)
        labels = np.squeeze(labels)
        #标签转换为下标0,1
        labels = ((labels+1)/2).astype(np.int)

        return data,labels
    
    def __getitem__(self,index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (data, label) where target{0,1}.
        """

        data = self.data[index]
        label = self.labels[index]

        return data,label

    def __len__(self):
        return len(self.data)


if __name__ =="__main__":
    # data,label = load_train_norm_img()
    # print(data.shape)

    myset = FaceVisionSet(train=True)
    import cv2
    for i in range(10):
        cv2.imshow(str(myset[i][1]), myset[i][0][0,:])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
