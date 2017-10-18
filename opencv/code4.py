import cv2
import numpy as np

img = cv2.imread('watch.jpg', cv2.IMREAD_COLOR)

px = img[20,20] # image is like a 2D matrix
px = [255,255,255]
print(img[20,20]) # does'nt work on actual image

img[20,20] = [0,0,0]
px = img[20,20]
print(px)

img[100:200,100:200] = [0,0,0] # region

print(img.shape)
print(img.size)
print(img.dtype)

img[100:150,100:150] = img[50:100,50:100] # like stamp tool, copy-paste

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()