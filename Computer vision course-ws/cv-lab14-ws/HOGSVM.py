import cv2
import numpy as np
import glob
from sklearn import svm
from sklearn.externals import joblib
import itertools
import imutils
from imutils.object_detection import non_max_suppression
import os

hog = cv2.HOGDescriptor('/home/alireza/Documents/cv-lab14/hog.xml')
train_data = []
train_labels = []
X = []

# reading positive samples
fnamesPos = glob.glob('/home/alireza/Documents/cv-lab14/pos/*.png')

for fname in fnamesPos:
    I1 = cv2.imread(fname)
    I1 = I1[3:-3, 3:-3, :]
    feature = hog.compute(I1)
    train_data.append(feature)
    train_labels.append(1)

# reading negative samples
fnamesNeg = glob.glob('/home/alireza/Documents/cv-lab14/neg/*.png')

for fname in fnamesNeg:
    I1 = cv2.imread(fname)
    # creating some random negative samples from images which don't contain any pedestrians.
    samples1 = np.random.randint(0, I1.shape[1] - 64, 10)
    samples2 = np.random.randint(0, I1.shape[0] - 128, 10)
    samples = zip(samples1, samples2)
    for sample in samples:
        I2 = I1[sample[1]:sample[1] + 128, sample[0]:sample[0] + 64, :]
        feature = hog.compute(I2)
        train_data.append(feature)
        train_labels.append(0)

X = np.asarray(train_data, dtype=np.float64)
X = np.reshape(X, (X.shape[0], X.shape[1]))
train_labels = np.asarray(train_labels)

# training the SVM classifier
file_name = 'saved_svm.sav'

classifier = svm.SwVC(kernel='poly', C=1.7, tol=1e-6, coef0=1.5, gamma='auto', max_iter=-1)

# if not os.path.isfile(file_name):
#     print('training the SVM')
#     classifier.fit(X, train_labels)
#     joblib.dump(classifier, file_name)
# else:
#     classifier = joblib.load(file_name)
#     print('saved classifier loaded')

os.path.isfile( file_name )
print( 'training the SVM' )
classifier.fit( X, train_labels )
joblib.dump( classifier, file_name )
# else:
#     classifier = joblib.load( file_name )
#     print( 'saved classifier loaded' )

# putting the final support vector params in correct order for feeding to our HoGDescriptor
supportVectors = []
supportVectors.append(np.dot(classifier.dual_coef_, classifier.support_vectors_)[0])
supportVectors.append([classifier.intercept_])
supportVectors = list(itertools.chain(*supportVectors))
hog.setSVMDetector(np.array(supportVectors, dtype=np.float64))

# testing
scale = 1.05#1.05
padding = (6,6)#(4, 4)
winStride = (4,4)#(8, 8)

fnamesTest = glob.glob('/home/alireza/Documents/cv-lab14/Test/*.png')
for fname in fnamesTest:
    I = cv2.imread(fname)
    I = imutils.resize(I, width=min(400, I.shape[1]))
    (rects, weights) = hog.detectMultiScale(I, winStride=winStride, padding=padding, scale=scale,
                                            useMeanshiftGrouping=True)
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    rects = non_max_suppression(rects, probs=None, overlapThresh=0.6)
    for (xA, yA, xB, yB) in rects:
        cv2.rectangle(I, (xA, yA), (xB, yB), (0, 255, 0), 2)
    cv2.imshow("detected pedestrians", I)
    cv2.waitKey(0)
