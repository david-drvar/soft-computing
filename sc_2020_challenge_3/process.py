# import libraries here
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Sklearn biblioteka sa implementiranim K-means algoritmom
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
#from imblearn.over_sampling import SMOTE, ADASYN

import os
from sklearn.svm import SVC  # SVM klasifikator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier  # KNN
from joblib import dump, load

import matplotlib

from imutils import face_utils
import argparse
import imutils
import dlib


def load_image(path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    return resize_image(img)


def resize_image(image):
    return cv2.resize(image, (200, 200), interpolation=cv2.INTER_NEAREST)


def display_image(image):
    plt.imshow(image, 'gray')
    plt.show()


def create_hog_descriptor(shape):
    nbins = 120  # broj binova 12 - 55%, 15 - 55.82%
    cell_size = (5, 5)  # broj piksela po celiji (3,3) - 55%
    block_size = (5, 5)  # broj celija po bloku (5,5) - 56%

    hog = cv2.HOGDescriptor(_winSize=(shape[1] // cell_size[1] * cell_size[1],
                                      shape[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)

    return hog


# transformisemo u oblik pogodan za scikit-learn
def reshape_data(input_data):
    nsamples, nx, ny = input_data.shape
    return input_data.reshape((nsamples, nx * ny))


def extract_faces(images):
    result = []

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    for image in images:
        #display_image(image)
        orig = image.copy()
        gray = image
        rects = detector(gray, 1)
        #display_image(gray)

        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)  # konverzija u NumPy niz

            cnt=[]
            for (x, y) in shape:
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
                cnt.append([x,y])
            cnt = np.array(cnt)

            x, y, w, h = cv2.boundingRect(cnt)
            #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            b = y + h
            a = x + w
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            if b > 200:
                b = 200
            if a > 200:
                a = 200
            #display_image(image)
            cropped_image = orig[y:b, x:a]
            #display_image(cropped_image)
            result.append(cropped_image)

    return result


def prepare_model(train_image_paths, train_image_labels):
    images = []
    for path in train_image_paths:
        images.append(load_image(path))

    shape = images[0].shape
    hog = create_hog_descriptor(shape)

    features = []
    for image in images:
        features.append(hog.compute(image))

    features = reshape_data(np.array(features))
    labels = np.array(train_image_labels)

    return create_model(features, labels)

def prepare_model_dlib_hog_combo(train_image_paths, train_image_labels):
    images = []
    for path in train_image_paths:
        images.append(load_image(path))

    images = extract_faces(images)

    resized_images = []
    for image in images:
        resized_images.append(resize_image(image))

    shape = resized_images[0].shape
    hog = create_hog_descriptor(shape)

    features = []
    a = 4

    for image in resized_images:
        a = image.copy()
        features.append(hog.compute(image))


    features = reshape_data(np.array(features))
    train_image_labels.pop()
    labels = np.array(train_image_labels)

    return create_model(features, labels)


def create_model(features, labels):
    clf_svm = SVC(kernel='linear', probability=True)
    #features_resampled, labels_resampled = SMOTE().fit_resample(labels, features)
    clf_svm.fit(features, labels)

    return clf_svm


def prepare_features_dlib(train_image_paths):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    features = []
    for image_path in train_image_paths:
        gray = load_image(image_path)
        rects = detector(gray, 1)

        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            features.append(shape)

    return features


def prepare_model_dlib(train_image_paths, train_image_labels):
    features = prepare_features_dlib(train_image_paths)

    features = reshape_data(np.array(features))
    train_image_labels.pop()
    labels = np.array(train_image_labels)

    return create_model(features, labels)


def load_model(path):
    try:
        return load(path)
    except Exception as e:
        return None


def save_model(model, path):
    dump(model, path)


def train_or_load_age_model(train_image_paths, train_image_labels):
    model = load_model('serialization_folder/age_model.joblib')
    if model == None:
        model = prepare_model_dlib_hog_combo(train_image_paths, train_image_labels)
        save_model(model, 'serialization_folder/age_model.joblib')
    return model


def train_or_load_gender_model(train_image_paths, train_image_labels):
    model = load_model('serialization_folder/gender_model.joblib')
    if model == None:
        model = prepare_model_dlib_hog_combo(train_image_paths, train_image_labels)
        save_model(model, 'serialization_folder/gender_model.joblib')
    return model


def train_or_load_race_model(train_image_paths, train_image_labels):
    model = load_model('serialization_folder/race_model.joblib')
    if model == None:
        model = prepare_model_dlib_hog_combo(train_image_paths, train_image_labels)
        save_model(model, 'serialization_folder/race_model.joblib')
    return model


def predict_age(trained_model, image_path):
    # todo combo dlib hog
    # img = cv2.resize(load_image(image_path), (200, 200), interpolation=cv2.INTER_NEAREST)
    # images = []
    # images.append(img)
    #
    # images = extract_faces(images)
    #
    # resized_images = []
    # for image in images:
    #     resized_images.append(resize_image(image))
    #
    # if resized_images == []:
    #     return 20
    #
    # hog = create_hog_descriptor(resized_images[0].shape)
    # age = trained_model.predict(reshape_data(np.array([hog.compute(resized_images[0])])))
    # return age[0]

    # todo basic hog
    img = cv2.resize(load_image(image_path), (200, 200), interpolation=cv2.INTER_NEAREST)
    hog = create_hog_descriptor(img.shape)
    age = trained_model.predict(reshape_data(np.array([hog.compute(img)])))
    return age[0]

    # todo dlib
    # features = prepare_features_dlib([image_path])
    # if features == []:
    #     return 0
    # age = trained_model.predict(reshape_data(np.array(features)))
    # return age[0]


def predict_gender(trained_model, image_path):
    # todo basic hog
    img = cv2.resize(load_image(image_path), (200, 200), interpolation=cv2.INTER_NEAREST)
    hog = create_hog_descriptor(img.shape)
    gender = trained_model.predict(reshape_data(np.array([hog.compute(img)])))
    return gender[0]

    # todo hog dlib combo
    # img = cv2.resize(load_image(image_path), (200, 200), interpolation=cv2.INTER_NEAREST)
    # images = []
    # images.append(img)
    #
    # images = extract_faces(images)
    #
    # resized_images = []
    # for image in images:
    #     resized_images.append(resize_image(image))
    #
    # if resized_images == []:
    #     return 1
    #
    # hog = create_hog_descriptor(resized_images[0].shape)
    # gender = trained_model.predict(reshape_data(np.array([hog.compute(resized_images[0])])))
    # return gender[0]

    # todo dlib
    # features = prepare_features_dlib([image_path])
    # if features == []:
    #     return 0
    # gender = trained_model.predict(reshape_data(np.array(features)))
    # return gender[0]


def predict_race(trained_model, image_path):
    # todo hog basic
    img = cv2.resize(load_image(image_path), (200, 200), interpolation=cv2.INTER_NEAREST)
    hog = create_hog_descriptor(img.shape)
    race = trained_model.predict(reshape_data(np.array([hog.compute(img)])))
    return race[0]

    # todo hog dlib combo
    # img = cv2.resize(load_image(image_path), (200, 200), interpolation=cv2.INTER_NEAREST)
    # images = []
    # images.append(img)
    #
    # images = extract_faces(images)
    #
    # resized_images = []
    # for image in images:
    #     resized_images.append(resize_image(image))
    #
    # if resized_images == []:
    #     return 3
    #
    # hog = create_hog_descriptor(resized_images[0].shape)
    # race = trained_model.predict(reshape_data(np.array([hog.compute(resized_images[0])])))
    # return race[0]

    # todo dlib
    # features = prepare_features_dlib([image_path])
    # if features == []:
    #     return 0
    # race = trained_model.predict(reshape_data(np.array(features)))
    # return race[0]
