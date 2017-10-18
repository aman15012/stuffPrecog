import cv2
import numpy as np

img = cv2.imread('watch.jpg',cv2.IMREAD_GRAYSCALE) # to read in grayscale could be IMREAD_COLOR etc
cv2.imshow('image',img)
cv2.waitKey(0) # wail till a key is pressed
cv2.destroyAllWindows()

