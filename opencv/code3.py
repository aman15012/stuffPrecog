import numpy as np
import cv2

img = cv2.imread('watch.jpg',cv2.IMREAD_COLOR)# read

#cv2.line(img,(0,0),(200,300),(255,255,255),50) #line
cv2.rectangle(img,(500,250),(1000,500),(0,0,255),15) # rectangle

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'YOLO',(-100,100), font, 6, (200,255,155), 13, cv2.LINE_AA)


cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()