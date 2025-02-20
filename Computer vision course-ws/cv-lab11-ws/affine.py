import cv2
import numpy as np

I = cv2.imread('/home/alireza/Documents/cv-lab11/karimi.jpg')

t = np.array([[30],
              [160]], dtype=np.float32)
A = np.array([[.7, 0.8],
              [-0.3, .6]], dtype=np.float32)
# A = np.array([[1.2, 0],
#               [0, 1.2]], dtype=np.float32) # acting like scale




M = np.hstack([A,t])

output_size = (I.shape[1], I.shape[0])
J = cv2.warpAffine(I,M,  output_size)

cv2.imshow('I',I)
cv2.waitKey(0)
cv2.imshow('J',J)
cv2.waitKey(0)
