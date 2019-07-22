import numpy as np
import cv2

I=cv2.imread("/home/alireza/Documents/cv-lab4/isfahan.jpg",cv2.IMREAD_GRAYSCALE)

I = I.astype(np.float)/255

sigma= 1

N = np.random.rand(*I.shape)*sigma

while True:

    N = np.random.rand( *I.shape ) * sigma
    J=I + N

    cv2.imshow("snow noise",J)
# press any key to exit
    key = cv2.waitKey(33)

    if key == ord('u'):
        sigma+=0.1
    elif key == ord('d'):
        sigma-=0.1
        if sigma < 0:
            sigma=0
    elif key == ord('q'):
        break
cv2.destroyAllWindows()