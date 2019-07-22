from skimage import exposure
from skimage import feature
import cv2
import numpy as np

fname = '/home/alireza/Documents/cv-lab13/01.jpg'
logo = cv2.imread(fname)
logo=cv2.cvtColor(logo,cv2.COLOR_BGR2GRAY)
(H, hogImage) = feature.hog(logo, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1",
                            visualise=True)
hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
hogImage = hogImage.astype("uint8")

cv2.imshow("HOG Image", hogImage)
cv2.waitKey(0)
