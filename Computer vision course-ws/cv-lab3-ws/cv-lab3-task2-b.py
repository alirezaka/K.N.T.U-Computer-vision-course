import cv2
import numpy as np
from matplotlib import pyplot as plt



def secDeriv(I):
    c=0
    k=0
    for i in range(0,254):
        if int(I.ravel()[i+2])-int(I.ravel()[i+1]) < abs(6):
            c+=1
            if c>32:
                return i-c
        # elif:
        #     int(I.ravel()[i+2])-int(I.ravel()[i+1]) < abs(6):
        #     k+=1
        #     if k>22:
        #         return i-k
        else:
            c=0
            k=0


if __name__=="__main__":


# fname = '/home/alireza/Documents/cv-lab3/crayfish.jpg'
#fname = '/home/alireza/Documents/cv-lab3/office.jpg'
#fname = '/home/alireza/Documents/cv-lab3/map.jpg'
# fname = '/home/alireza/Documents/cv-lab3/train.jpg'
 fname = '/home/alireza/Documents/cv-lab3/branches.jpg'

 I = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)

 f, axes = plt.subplots(2, 3)

 axes[0,0].imshow(I, 'gray', vmin=0, vmax=255)
 axes[0,0].axis('off')

 axes[1,0].hist(I.ravel(),256,[0,256]);


# a=100  #crayfish
# b=175  #crayfish


#a=150  #office
#b=200  #office


# a=150  #map
# b=220  #map

# a=75  #train
# b=225  #train

 a=150  #branches
 b=220  #branches


 J = (I-a) * 255.0 / (b-a)
 J[J < 0] = 0
 J[J > 255] = 255
 J.astype(np.uint8)

 axes[0,1].imshow(J, 'gray', vmin=0, vmax=255)
 axes[0,1].axis('off')

 axes[1,1].hist(J.ravel(),256,[0,256])

# print(I)


 K = cv2.equalizeHist(I)

 axes[0,2].imshow(K, 'gray', vmin=0, vmax=255)
 axes[0,2].axis('off')

 axes[1,2].hist(K.ravel(),256,[0,256])


 plt.show()

 print(secDeriv(I.ravel()))


