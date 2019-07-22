import cv2
import numpy as np
from matplotlib import pyplot as plt

fname = '/home/alireza/Documents/cv-lab3/crayfish.jpg'
#fname = 'office.jpg'

I = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)

f, axes = plt.subplots(2, 3)

axes[0,0].imshow(I, 'gray', vmin=0, vmax=255)
axes[0,0].axis('off')

axes[1,0].hist(I.ravel(),256,[0,256]);


a=100
b=175


J = (I-a) * 255.0 / (b-a)
J[J < 0] = 0
J[J > 255] = 255
J.astype(np.uint8)

axes[0,1].imshow(J, 'gray', vmin=0, vmax=255)
axes[0,1].axis('off')

axes[1,1].hist(J.ravel(),256,[0,256])


plt.show()