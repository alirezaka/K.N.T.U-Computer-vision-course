import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.externals import joblib
import os
import random

dataset = '/home/alireza/Documents/cv-lab13/dataset/train/{}/'

train_images_list1 = os.listdir(dataset.format('1'))
train_images_list0 = os.listdir(dataset.format('0'))

train_images_list1.sort()
train_images_list0.sort()

train_labels = [1 for i in range(len(train_images_list1))]
train_labels.extend([0 for i in range(len(train_images_list0))])

input_data = []

for addr in np.ravel([train_images_list0, train_images_list1]):
    I = cv2.imread(os.path.join('/home/alireza/Documents/cv-lab13/dataset/train/{}/'.format(str(addr[2])), addr))
    input_data.append(I.ravel())


classifier = SVC(gamma='auto')
file_name = 'saved_svm.sav'

if not os.path.isfile(file_name):
    classifier.fit(input_data, train_labels)
    joblib.dump(classifier, file_name)

else:
    classifier = joblib.load(file_name)

test_dir=dataset = '/home/alireza/Documents/cv-lab13/dataset/test'
test_images_list = os.listdir(test_dir)
test_images_list.sort()

test_labels = [int(addr[2]) for addr in (test_images_list)]

test_data = []

for addr in (test_images_list):
    J = cv2.imread(os.path.join('/home/alireza/Documents/cv-lab13/dataset/test'.format(str(addr[2])), addr))
    # cv2.imshow('J',J)
    # cv2.waitKey(0)
    test_data.append(J.ravel())


idx = [random.randint(0, len(test_data)-1) for i in range(10)]

test_input = [test_data[i] for i in idx]

test_labels = [test_labels[i] for i in idx]
results = classifier.predict(test_input)
print('predictions: ', results)
print("train lables: ",train_labels)
print("test lables: ",test_labels)
print("Accuracy: ",(np.sum(results==test_labels)/len(results))*100,"%")
print("-------------------------")
idx = [random.randint(0, len(input_data)-1) for i in range(10)]
test_input = [input_data[i] for i in idx]
test_labels = [train_labels[i] for i in idx]
results = classifier.predict(test_input)
print('predictions: ', results)
print("train lables: ",train_labels)
print("test lables: ",test_labels)
print("Accuracy: ",(np.sum(results==test_labels)/len(results))*100,"%")