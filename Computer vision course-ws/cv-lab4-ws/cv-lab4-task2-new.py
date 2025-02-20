import numpy as np
import cv2

I = cv2.imread('/home/alireza/Documents/cv-lab4/isfahan.jpg').astype(np.float64) / 255;

# display the original image
cv2.imshow('original',I)
cv2.waitKey()

# creating a box filter
m = 7 # choose filter size

# create an m by m box filter
F = np.ones((m,m), np.float64)/(m*m)
print(F)

# Now, filter the image
J = cv2.filter2D(I,-1, F)
cv2.imshow('blurred',J)
cv2.waitKey()

J=cv2.blur(I,(7,7))
cv2.imshow('blurred_using_cv2.blur',J)
cv2.waitKey()

cv2.destroyAllWindows()
