import cv2
from skimage import feature as ft
import numpy as np
'''
2002/08/11/big/img_591
1
123.583300 85.549500 1.265839 269.693400 161.781200  1
'''
imgs = cv2.imread("img_591.jpg")
print(imgs.shape)
centers = (269,161)
axess =  (123,85)
angles = 1.265839/np.pi*180.0

cv2.ellipse(imgs, centers,axess,angles,0,360,(0,0,255), 3)

cv2.imshow("img_591",imgs)

# Block size = 2 x 2
# Cell size = 16 x 16
# Number of bins = 9
# Block overlap = 1 x 1
img = cv2.resize(imgs, (96, 96))

hog_ft,hog_img = ft.hog(img,
            orientations=9,
            pixels_per_cell=(16,16),
            cells_per_block=(2,2),
            transform_sqrt= True,
            feature_vector=True,
            visualize=True)

cv2.imshow("hog",hog_img)
print(hog_ft.shape)
cv2.waitKey(0)

cv2.destroyAllWindows()
