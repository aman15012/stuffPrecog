import numpy as np
import cv2

cap = cv2.VideoCapture(0) #capture video from webcam

# To record and save the video
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

while(True):
    ret, frame = cap.read() # frame has each frame being returned
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert frame to grayscale
 
    cv2.imshow('frame',gray) # simultaneously show the converted image
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break # break if 'q'is pressed # not working

cap.release()
# out.release()
cv2.destroyAllWindows()