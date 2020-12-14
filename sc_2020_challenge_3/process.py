# import libraries here
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Sklearn biblioteka sa implementiranim K-means algoritmom
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

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


# transformisemo u oblik pogodan za scikit-learn
def reshape_data(input_data):
    nsamples, nx, ny = input_data.shape
    return input_data.reshape((nsamples, nx * ny))


def create_hog_descriptor(shape):
    nbins = 16  # broj binova 12 - 55%
    cell_size = (3, 3)  # broj piksela po celiji (3,3) - 55%
    block_size = (5, 5)  # broj celija po bloku (5,5) - 56%

    hog = cv2.HOGDescriptor(_winSize=(shape[1] // cell_size[1] * cell_size[1],
                                      shape[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)

    return hog


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


def create_model(features, labels):
    clf_svm = SVC(kernel='linear', probability=True)
    clf_svm.fit(features, labels)

    return clf_svm


def prepare_features_dlib(train_image_paths):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    features = []
    for image_path in train_image_paths:
        if image_path == '.\\dataset\\train\\image_12.jpg':
            # temp = []
            # temp.append([0,0])
            # features.append(temp)
            # continue
            a = 4
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
    """
    Procedura prima listu putanja do fotografija za obucavanje (dataset se sastoji iz razlicitih fotografija), liste
    labela za svaku fotografiju iz prethodne liste, kao i putanju do foldera u koji treba sacuvati model nakon sto se
    istrenira (da ne trenirate svaki put iznova)

    Procedura treba da istrenira model i da ga sacuva u folder "serialization_folder" pod proizvoljnim nazivom

    Kada se procedura pozove, ona treba da trenira model ako on nije istraniran, ili da ga samo ucita ako je prethodno
    istreniran i ako se nalazi u folderu za serijalizaciju

    :param train_image_paths: putanje do fotografija za obucavanje
    :param train_image_labels: labele za sve fotografije iz liste putanja za obucavanje
    :return: Objekat modela
    """
    # TODO - Istrenirati model ako vec nije istreniran, ili ga samo ucitati iz foldera za serijalizaciju

    model = load_model('serialization_folder/age_model.joblib')
    if model == None:
        model = prepare_model(train_image_paths, train_image_labels)
        save_model(model, 'serialization_folder/age_model.joblib')
    return model


def train_or_load_gender_model(train_image_paths, train_image_labels):
    """
    Procedura prima listu putanja do fotografija za obucavanje (dataset se sastoji iz razlicitih fotografija), liste
    labela za svaku fotografiju iz prethodne liste, kao i putanju do foldera u koji treba sacuvati model nakon sto se
    istrenira (da ne trenirate svaki put iznova)

    Procedura treba da istrenira model i da ga sacuva u folder "serialization_folder" pod proizvoljnim nazivom

    Kada se procedura pozove, ona treba da trenira model ako on nije istraniran, ili da ga samo ucita ako je prethodno
    istreniran i ako se nalazi u folderu za serijalizaciju

    :param train_image_paths: putanje do fotografija za obucavanje
    :param train_image_labels: labele za sve fotografije iz liste putanja za obucavanje
    :return: Objekat modela
    """
    # TODO - Istrenirati model ako vec nije istreniran, ili ga samo ucitati iz foldera za serijalizaciju

    model = load_model('serialization_folder/gender_model.joblib')
    if model == None:
        model = prepare_model(train_image_paths, train_image_labels)
        save_model(model, 'serialization_folder/gender_model.joblib')
    return model


def train_or_load_race_model(train_image_paths, train_image_labels):
    """
    Procedura prima listu putanja do fotografija za obucavanje (dataset se sastoji iz razlicitih fotografija), liste
    labela za svaku fotografiju iz prethodne liste, kao i putanju do foldera u koji treba sacuvati model nakon sto se
    istrenira (da ne trenirate svaki put iznova)

    Procedura treba da istrenira model i da ga sacuva u folder "serialization_folder" pod proizvoljnim nazivom

    Kada se procedura pozove, ona treba da trenira model ako on nije istraniran, ili da ga samo ucita ako je prethodno
    istreniran i ako se nalazi u folderu za serijalizaciju

    :param train_image_paths: putanje do fotografija za obucavanje
    :param train_image_labels: labele za sve fotografije iz liste putanja za obucavanje
    :return: Objekat modela
    """
    # TODO - Istrenirati model ako vec nije istreniran, ili ga samo ucitati iz foldera za serijalizaciju

    model = load_model('serialization_folder/race_model.joblib')
    if model == None:
        model = prepare_model(train_image_paths, train_image_labels)
        save_model(model, 'serialization_folder/race_model.joblib')
    return model


def predict_age(trained_model, image_path):
    """
    Procedura prima objekat istreniranog modela za prepoznavanje godina i putanju do fotografije na kojoj
    se nalazi novo lice sa koga treba prepoznati godine.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param trained_model: <Model> Istrenirani model za prepoznavanje godina
    :param image_path: <String> Putanja do fotografije sa koje treba prepoznati godine lica
    :return: <Int> Prediktovanu vrednost za goinde  od 0 do 116
    """
    age = 0
    img = cv2.resize(load_image(image_path), (200, 200), interpolation=cv2.INTER_NEAREST)
    hog = create_hog_descriptor(img.shape)
    age = trained_model.predict(reshape_data(np.array([hog.compute(img)])))
    return age[0]

    # features = prepare_features_dlib([image_path])
    # if features == []:
    #     return 0
    # age = trained_model.predict(reshape_data(np.array(features)))
    # return age[0]


def predict_gender(trained_model, image_path):
    """
    Procedura prima objekat istreniranog modela za prepoznavanje pola na osnovu lica i putanju do fotografije na kojoj
    se nalazi novo lice sa koga treba prepoznati pol.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param trained_model: <Model> Istrenirani model za prepoznavanje karaktera
    :param image_path: <String> Putanja do fotografije sa koje treba prepoznati ekspresiju lica
    :return: <Int>  Prepoznata klasa pola (0 - musko, 1 - zensko)
    """

    gender = 0
    img = cv2.resize(load_image(image_path), (200, 200), interpolation=cv2.INTER_NEAREST)
    hog = create_hog_descriptor(img.shape)
    gender = trained_model.predict(reshape_data(np.array([hog.compute(img)])))
    return gender[0]

    # features = prepare_features_dlib([image_path])
    # if features == []:
    #     return 0
    # gender = trained_model.predict(reshape_data(np.array(features)))
    # return gender[0]


def predict_race(trained_model, image_path):
    """
    Procedura prima objekat istreniranog modela za prepoznavanje rase lica i putanju do fotografije na kojoj
    se nalazi novo lice sa koga treba prepoznati rasu.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param trained_model: <Model> Istrenirani model za prepoznavanje karaktera
    :param image_path: <String> Putanja do fotografije sa koje treba prepoznati ekspresiju lica
    :return: <Int>  Prepoznata klasa (0 - Bela, 1 - Crna, 2 - Azijati, 3- Indijci, 4 - Ostali)
    """

    race = 0
    img = cv2.resize(load_image(image_path), (200, 200), interpolation=cv2.INTER_NEAREST)
    hog = create_hog_descriptor(img.shape)
    race = trained_model.predict(reshape_data(np.array([hog.compute(img)])))
    return race[0]

    # features = prepare_features_dlib([image_path])
    # if features == []:
    #     return 0
    # race = trained_model.predict(reshape_data(np.array(features)))
    # return race[0]
