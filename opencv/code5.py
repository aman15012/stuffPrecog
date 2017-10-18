import cv2
import numpy as np


img1 = cv2.imread('3D-Matplotlib.png')
img2 = cv2.imread('mainsvmimage.png')

add = img1+img2 # same size, simples overlaps images

weighted = cv2.addWeighted(img1, 0.6, img2, 0.4, 0) # actually does a weighed sunm of pixels

# cv2.imshow('add',add)
cv2.imshow('weighted',weighted)
cv2.waitKey(0)
cv2.destroyAllWindows()

