# -*- coding:utf-8 -*-
# python3  
import os

import cv2
import numpy as np
import pandas as pd
from skimage import feature as ft


AnnotationFolder = "FDDB-folds"
OriginalPicsFolder = "originalPics"

ImgAnnotation = "FDDB-fold-01-ellipseList.txt"
ImgList = "FDDB-fold-01.txt"

VisDir = "vis_data/vis_data"
np_hog_set_dir = "data/hog/"
np_img_set_dir = "data/img/"

class Face(object):
    #存储单个脸的完整信息
    #major_axis_radius minor_axis_radius angle center_x center_y 1
    #半长轴(一般是脸的竖直方向)，半短轴(一般脸的水平方向)，
    #角度((-pi/2 to pi/2)单位是弧度，angle等于 长轴 和 x轴正半轴 的夹角，以逆时针为正)
    def __init__(self,major_axis_radius,minor_axis_radius,angle,center_x,center_y):
        
        # below all float
        self.major_axis = major_axis_radius
        self.minor_axis = minor_axis_radius
        self.angle = angle
        self.center_x = center_x
        self.center_y = center_y

        #below all integer
        #调整后的图片边框
        self.left = 0
        self.right = 0
        self.top = 0
        self.bottom =0
        self.width =0
        self.height = 0 
        self.cal_bound_box()
    
    def cal_bound_box(self):
        #简单计算调整后的像素边框 无旋转
        y_radius = self.major_axis*4.0/3.0
        x_radius = self.minor_axis*4.0/3.0
        self.top = int(self.center_y - y_radius)
        self.bottom = int(self.center_y + y_radius)
        self.left = int(self.center_x - x_radius)
        self.right = int(self.center_x + x_radius)
        self.width = self.right-self.left
        self.height = self.bottom - self.top


    
    #可选构造函数
    @classmethod
    def from_string(cls,face_ellipse_string):
        # face_ellipse_string should be a string like below
        # "123.583300 85.549500 1.265839 269.693400 161.781200  1"
        face_ellipse = face_ellipse_string.split(" ")
        assert len(face_ellipse)!=6, "Wrong Data Format for Face Ellipse!"

        major_axis = float(face_ellipse[0])
        minor_axis = float(face_ellipse[1])
        angle = float(face_ellipse[2])
        center_x = float(face_ellipse[3])
        center_y = float(face_ellipse[4])

        return cls(major_axis,minor_axis,angle,center_x,center_y)

class ImgInfo(object):
    #存储单张图片的完整信息
    #img 处理步骤
    # read_img_info -> read_pic -> auto_padding -> cut_face_samples and resize 
    #                               |               |-> trans to hog feature ->save samples
    #                               |-> draw all thing and show ->save
    def __init__(self, path, face_num, origin_path):
        self.path = path #图片完整路径
        self.face_num = face_num  # 脸的数量
        self.origin_path = origin_path #原始路径
        self.face_list = [] #脸的信息

        self.face_imgs = [] #脸的图片的信息
        self.face_labels = []  # 脸的图片的label的信息
        self.face_hogs = [] # 脸的HOG信息
        self.face_hogs_img = []  # 脸的HOG信息图片
    
    def __str__(self):
        info = "origin path: [{}]\nface_num: [{}]".format(self.path, self.face_num)
        return info

    def add_face_info(self, face):
        self.face_list.append(face)
    
    def save_face_img(self):
        # data/2002/08/11/big/img_591
        prefix = VisDir + "-".join(self.origin_path.split("/"))
        print(prefix)
        for i in range(len(self.face_imgs)):
            face_dir = ""
            if(self.face_labels[i]==-1):
                face_dir = prefix + "-" + str(i) + "-neg.jpg"
            elif(self.face_labels[i]==1):
                face_dir = prefix + "-" + str(i) + "-pos.jpg"
            hog_dir = prefix + "-" + str(i) + "-hog.jpg"

            #### 灰度数值缩放
            hog_img = self.face_hogs_img[i]*255
            hog_img = hog_img.astype(np.uint8)

            cv2.imwrite(face_dir, self.face_imgs[i])
            cv2.imwrite(hog_dir, hog_img)

    def save_expand_img(self,img):
        # data/2002/08/11/big/img_591
        prefix = VisDir + "-".join(self.origin_path.split("/"))
        img_dir = prefix+"-expand.jpg"
        cv2.imwrite(img_dir, img)

    # def save_hog_img(self, img):
    #     # data/2002/08/11/big/img_591
    #     prefix = "vis_data/" + "-".join(self.path.split("/"))
    #     img_dir = prefix+"-hog.jpg"
    #     cv2.imwrite(img_dir, img)
    def auto_padding(self,img):
        padding = int(max(img.shape))

        for face in self.face_list:
            max_h = abs(face.center_y)+face.height
            max_w = abs(face.center_x)+face.width
            padding = int(max([padding,max_h,max_w]))

        img_padding = cv2.copyMakeBorder(
            img, padding, padding, padding, padding, cv2.BORDER_REPLICATE)
        return img_padding,padding


    def get_all_face(self,img, padding = 0, resize=True):
        """
        cut out all face(positive and negtive) in the the picture with padding
        
        Args:
            img: cv2 BGR image already padding
            padding: padding size around picture
            resize: whether resize face to 96px*96px
        Returns:
            face_imgs: 
            face_lables:
        """
        #切出所有正负样本的框
        for face in self.face_list:
            for s_w in [-1,0,1]:#横向移位数
                for s_h in [-1,0,1]: #纵向移位数
                    
                    label = -1
                    if s_w==0 and s_h==0: 
                        label = +1
                    w = int(face.width*1.0/3.0)*s_w 
                    h = int(face.height*1.0/3.0)*s_h 
                    l, t = (face.left+padding+w, face.top+padding+h)
                    r, b = (face.right+padding+w, face.bottom+padding+h)
                    face_img = img[t:b, l:r]
                    if(resize):
                        # print(face_img.shape)
                        face_img=cv2.resize(face_img,(96,96))

                    hog_ft, hog_img = ft.hog(face_img,
                                             orientations=9,
                                             pixels_per_cell=(16, 16),
                                             cells_per_block=(2, 2),
                                             transform_sqrt=True,
                                             feature_vector=True,
                                             visualize=True)
                    self.face_imgs.append(face_img)
                    self.face_labels.append(label)
                    self.face_hogs.append(hog_ft)
                    self.face_hogs_img.append(hog_img)


        return self.face_imgs, self.face_labels,self.face_hogs,self.face_hogs_img

        
        

    def draw_face_ellipse(self,img, shift=0):
        """draw all the face in the img

        Args:
            img: cv image already padding
            shift: use to shift face box along x-aixs and y-aixs
        Returns:
            no returns, only modify the input img
        """
        for face in self.face_list:
            face_center = (int(face.center_x+shift),
                           int(face.center_y+shift))
            face_axes = (int(face.major_axis), int(face.minor_axis))
            face_angle = face.angle/np.pi*180.0  #详情见ellipse函数说明
            cv2.ellipse(img,face_center,face_axes,face_angle,0,360,(255,0,0),2)
    
    def draw_face_rectangle(self, img, shift=0):
        # draw all the rectangle of face
        for face in self.face_list:
            lt = (face.left+shift, face.top+shift)
            rb = (face.right+shift, face.bottom+shift)
            cv2.rectangle(img, lt, rb, (0, 255, 0),2)
    

    def draw_all_face_box(self,img,padding=0):
        """
        draw all face(positive and negtive) in the the picture with padding
        Args:
            img: cv2 BGR image already padding
            padding: padding size around picture

        Returns:
            on returns, only modify input image
        """

        #####画出脸的所有框

        #画椭圆
        self.draw_face_ellipse(img,shift = padding)

        #画出所有正负样本的框
        #负样本
        for face in self.face_list:
            for s_w in [-1,0,1]:#横向移位数
                for s_h in [-1,0,1]: #纵向移位数
                    if s_w==0 and s_h==0: 
                        continue
                    w = int(face.width*1.0/3.0)*s_w + int(face.width/40.0)*s_h
                    h = int(face.height*1.0/3.0)*s_h + int(face.height/40.0)*s_w
                    lt = (face.left+padding+w, face.top+padding+h)
                    rb = (face.right+padding+w, face.bottom+padding+h)
                    cv2.rectangle(img, lt, rb, (50, 50, 50), 2)
        #正样本
        self.draw_face_rectangle(img,padding)


def show_all_face(img_info_lists):
    #测试用，画出所有的椭圆框并显示
    for img_info in img_info_lists:
        img = cv2.imread(img_info.path)
        img_info.draw_face_ellipse(img)
        img_info.draw_face_rectangle(img)
        cv2.imshow(str(img_info.face_num), img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


class DataPreprocessing(object):
    def __init__(self):
        self.img_info_lists = []
        self.all_face_labels = []
        self.all_face_hogs = []
    
    def data_format(self,order, vis = False):
        # order is integer, the order of n-th data-set
        global VisDir
        VisDir = "vis_data/vis_data{:0>2d}/".format(order)
        # print(VisDir)
        # return
        annotation_path = os.path.join(
            AnnotationFolder, "FDDB-fold-{:0>2d}-ellipseList.txt".format(order))
        
        self._read_img_info(annotation_path)
        self._get_all_face_data(vis=vis)
        np.save(np_hog_set_dir +
                "face_hogs{:0>2d}.npy".format(order), np.asarray(self.all_face_hogs))
        np.save(np_hog_set_dir +
                "face_labels{:0>2d}.npy".format(order), np.asarray(self.all_face_labels))
        self.img_info_lists = []
        self.all_face_labels = []
        self.all_face_hogs = []

    def _read_img_info(self, img_anno_path):
        #读取图片信息列表
        with open(img_anno_path, "r") as img_annos:
            while True:  # 读取一张图片的所有信息
                path = img_annos.readline().strip('\n')
                if("" == path):
                    break
                origin_path =path
                path = os.path.join(OriginalPicsFolder, (path+".jpg"))  # 图片路径
                face_num = int(img_annos.readline())  # 图片数量

                img_info = ImgInfo(path, face_num, origin_path)  # 图片信息

                for i in range(face_num):  # 读取所有脸
                    face_ellipse = img_annos.readline().strip('\n')
                    face = Face.from_string(face_ellipse)  # 字符串转为脸
                    img_info.add_face_info(face)  # 图片加入脸
                    # print(img_info)
                # 一张图片所有信息读取完毕
                self.img_info_lists.append(img_info)  # 图片信息加入列表
        #图片信息读取完毕
        ##############################

    def _get_all_face_data(self,vis=False):
        ############################
        # 图片样本提取
        for img_info in self.img_info_lists:
            print(img_info)
            # if(img_info.path != "originalPics/2002/07/25/big/img_1047.jpg"):
            #     continue
            img = cv2.imread(img_info.path)

            ##expand src img by padding
            # special img originalPics/2002/07/25/big/img_1047.jpg

            img_padding, padding = img_info.auto_padding(img)

            # 切割出所有脸
            face_imgs, face_labels,face_hogs,face_hogs_img = img_info.get_all_face(img_padding, padding)

            if vis:
                img_info.save_face_img()
                img_info.draw_all_face_box(img_padding,padding)
                img_info.save_expand_img(img_padding)

            self.all_face_labels.extend(face_labels)
            self.all_face_hogs.extend(face_hogs)
    
def img_to_np(start,end):
    """
    加载切割好后的给定图片数据集[start to end],包含start和end指定数据集
    保存为np格式
    Args:
        start(int):数据集的开始序号
        end(int):数据集的结束序号
    
    Returns：
        img_data: 灰度图片数据np矩阵(n x height x width )
        img_label: 灰度图片标签{-1,+1}(n x 1)
    """
    label_dict = {
        "neg": -1,
        "pos": 1
    }

    for order in range(start, end+1, 1):
        img_dir = "./vis_data/vis_data{:0>2d}/".format(order)
        assert os.path.exists(img_dir), "[error: no path "+img_dir+" !]"

        print("formating: ", img_dir)
        img_data = []
        img_label = []

        img_names = os.listdir(img_dir)
        for names in img_names:
            if names[-7:-4] in label_dict.keys():
                #获取图片完整路径
                img_path = os.path.join(img_dir, names)

                #获取标签和图像
                label = label_dict[names[-7:-4]]
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                #加入数据集
                img_data.append(img)
                img_label.append(label)

        img_data = np.asarray(img_data)
        img_label = np.asarray(img_label)
        np.save(np_img_set_dir+"face_imgs{:0>2d}.npy".format(order),img_data)
        np.save(np_img_set_dir +
                "face_labels{:0>2d}.npy".format(order), img_label)


if __name__ == "__main__":
    # my_data = DataPreprocessing()
    # for i in range(1,11,1):
    #     my_data.data_format(i, vis=True)
    img_to_np(1,10)
