import numpy as np
import cv2
from sklearn.svm import SVC
import joblib
import os
import random
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from concurrent import futures

dataset = './dataset/train/'
train_labels = []
train_imgs_list = []

files = os.listdir(dataset)
files.sort()

for idx, file in enumerate(files[:3]):
    train_images_list = os.listdir(dataset + file)

    train_images_list.sort()

    train_labels.extend([idx for _ in range(len(os.listdir(dataset + file)))])

    train_imgs_list.append(train_images_list)

input_data = []


def train_processing(idx, file):
    print("Train Processing:" + file)
    temp = []

    for addr in train_imgs_list[idx]:
        I = cv2.imread(os.path.join(dataset + file, addr))

        temp.append(I.ravel())
    return temp


with futures.ProcessPoolExecutor() as executor:
    indices = np.arange(len(train_imgs_list))
    results = executor.map(train_processing, indices, files[:3])

    for result in results:
        input_data.extend(result)

scaler = StandardScaler()
X = scaler.fit_transform(input_data)

C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

max_score = 0
C = 0
gamma = 0


def grid_search(gamma_range, C_range):
    print("Grid search started...")
    param_grid = dict(gamma=[gamma_range], C=[C_range])
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
    grid.fit(X, train_labels)
    return grid.best_params_, grid.best_score_


with futures.ProcessPoolExecutor() as executor:
    results = executor.map(grid_search, gamma_range, C_range)

    for result in results:
        print(f"Parameters are {result[0]} with a score of {result[1] * 100: 3.3f} %")
        if result[1] > max_score:
            max_score = result[1]
            C = result[0]['C']
            gamma = result[0]['gamma']

classifier = SVC(C=C, gamma=gamma)  # C = 100, gamma = 1e-5
file_name = 'saved_svm.sav'

classifier.fit(X, train_labels)

joblib.dump(classifier, file_name)

test_dir = './dataset/test/'
files = os.listdir(test_dir)
files.sort()

test_labels = []
test_imgs_list = []

for idx, file in enumerate(files[:3]):
    test_images_list = os.listdir(test_dir + file)

    test_images_list.sort()

    test_labels.extend([idx for _ in range(len(os.listdir(test_dir + file)))])

    test_imgs_list.append(test_images_list)

test_data = []


def test_processing(idx, file):
    print("Test Processing:" + file)
    temp = []

    for addr in test_imgs_list[idx]:
        J = cv2.imread(os.path.join(test_dir + file, addr))
        # J = cv2.resize(J, (32, 32))
        temp.append(J.ravel())
    return temp


with futures.ProcessPoolExecutor() as executor:
    indices = np.arange(len(test_imgs_list))
    results = executor.map(test_processing, indices, files[:3])

    for result in results:
        test_data.extend(result)

print("-------------------------")

idx = [random.randint(0, len(test_data) - 1) for i in range(20)]
test_input = [test_data[i] for i in idx]
test_labels = [test_labels[i] for i in idx]
test_input = scaler.fit_transform(test_input)
results = classifier.predict(test_input)
print('predictions: ', results)
print("train lables: ", list(set(train_labels)))
print("test lables: ", test_labels)
print("Accuracy: ", (np.sum(results == test_labels) / len(results)) * 100, "%")

print("-------------------------")

idx = [random.randint(0, len(input_data) - 1) for i in range(10)]
test_input = [input_data[i] for i in idx]
test_labels = [train_labels[i] for i in idx]
test_input = scaler.fit_transform(test_input)
results = classifier.predict(test_input)
print('predictions: ', results)
print("train lables: ", list(set(train_labels)))
print("test lables: ", test_labels)
print("Accuracy: ", (np.sum(results == test_labels) / len(results)) * 100, "%")
