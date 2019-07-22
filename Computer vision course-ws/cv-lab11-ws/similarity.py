import cv2
import numpy as np

I = cv2.imread('/home/alireza/Documents/cv-lab11/karimi.jpg')

tx = 100
ty = 60

th =  20 # angle of rotation (degrees)
th *= np.pi / 180 # convert to radians

s = 0.2# scale factor
# s = 2# scale factor

M = np.array([[s*np.cos(th),-s*np.sin(th),tx],
              [s*np.sin(th), s*np.cos(th),ty]])

if s > 1:
    output_size = (I.shape[1]*s,I.shape[0]*s)
else:
    output_size = (I.shape[1] , I.shape[0] )
J = cv2.warpAffine(I,M,  output_size)
print(J.shape)
# J.resize((I.shape[0]*s,I.shape[1]*s,3))
cv2.imshow('I',I)
cv2.waitKey(0)
# print(J.shape)
cv2.imshow('J',J)
cv2.waitKey(0)
