import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.externals import joblib
import os
import random


dataset = '/home/alireza/Documents/cv-lab13/dataset/train/{}/'

for i in range(0,10):

    train_images_list = os.listdir(dataset.format(str(i)))
    input_data = []
    for addr in (train_images_list):

        I = cv2.imread(os.path.join(dataset.format(str(addr[2])), addr))
        input_data.append( I )

    idx = [random.randint( 0, len(input_data)-1 ) for i in range( 4 )]
    for k in range(0,4):
         cv2.imshow("I",input_data[idx[k]])
         q =  cv2.waitKey(0)
         if q==ord('q'):
          exit(0)
