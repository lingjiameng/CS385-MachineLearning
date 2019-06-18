import time
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import matplotlib.pyplot as plt
import sklearn.svm
import cv2
from skimage import feature as ft

import dataloader as dl
import mymodel



def find_face(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    print(img.shape)

    stride = 10

    win_min = 100
    win_max = 400
    win_step = 20

    mylog = mymodel.load_model("save/log.model")
    mysvm = mymodel.load_model("save/svm.model")
    mylda = mymodel.load_model("save/lda.model")
    mycnn = torch.load("save/cnn.pkl",map_location="cpu")


    def is_face(hog_ft):
        models = [mylog,mysvm, mylda]
        for m in models:
            if np.squeeze(m.predict(hog_ft)) !=1:
                return False
        return True


    for win in range(win_max,win_min,  -win_step):
        for i in range(0,img.shape[0],10+win//5):
            for j in range(0, img.shape[1], 10+win//5):
                if (i+win) > img.shape[0] or (j+win)>img.shape[1]:
                    continue
                face_img = img[i:i+win,j:j+win]
                face_img = cv2.resize(face_img,(96,96))
                hog_ft, hog_img = ft.hog(face_img,
                            orientations=9,
                            pixels_per_cell=(16, 16),
                            cells_per_block=(2, 2),
                            transform_sqrt=True,
                            feature_vector=True,
                            visualize=True)
                # print(hog_ft.shape)
                hog_ft = hog_ft.reshape(-1, 900)

                if not is_face(hog_ft):
                    continue

                face_ = face_img[np.newaxis, np.newaxis, :]
                face_ = face_.astype(np.float32)
                info = str(mycnn.predict(torch.tensor(face_)))

                lt = (j, i)
                rb = (j+win, i+win)
                cv2.rectangle(img, lt, rb, (0, 255, 0), 2)

                # cv2.imshow("face"+info,face_img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
    cv2.imshow("face",img)
    cv2.imwrite(filepath+"-res.png",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


parser = argparse.ArgumentParser(
    description='enter file dir')

parser.add_argument("--img", default="img_591.jpg", type=str,
                    help="enter image file path")

if __name__ == "__main__":
    args = parser.parse_args()
    assert os.path.exists(args.img), "error! no img path!"+args.img
    find_face(args.img)
