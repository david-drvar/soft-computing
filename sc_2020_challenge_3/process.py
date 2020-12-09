# import libraries here
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Sklearn biblioteka sa implementiranim K-means algoritmom
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN


import os
from sklearn.svm import SVC # SVM klasifikator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier # KNN
from joblib import dump, load

import matplotlib

from imutils import face_utils
import argparse
import imutils
import dlib

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)

def display_image(image):
    plt.imshow(image, 'gray')
    plt.show()


# transformisemo u oblik pogodan za scikit-learn
def reshape_data(input_data):
    nsamples, nx, ny = input_data.shape
    return input_data.reshape((nsamples, nx*ny))

def create_hog_descriptor(shape):
    nbins = 9  # broj binova
    cell_size = (8, 8)  # broj piksela po celiji
    block_size = (3, 3)  # broj celija po bloku

    hog = cv2.HOGDescriptor(_winSize=(shape[1] // cell_size[1] * cell_size[1],
                                      shape[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)

    return hog

def generate_model(train_image_paths, train_image_labels):
    images = []
    for path in train_image_paths:
        images.append(cv2.resize(load_image(path), (200,200), interpolation=cv2.INTER_NEAREST))

    shape = images[0].shape
    hog = create_hog_descriptor(shape)

    features = []
    for image in images:
        features.append(hog.compute(image))

    features = reshape_data(np.array(features))
    labels = np.array(train_image_labels)

    #SVM
    clf_svm = SVC(kernel='linear', probability=True)
    clf_svm.fit(features, labels)

    return clf_svm


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

    # # inicijalizaclija dlib detektora (HOG)
    # detector = dlib.get_frontal_face_detector()
    # # ucitavanje pretreniranog modela za prepoznavanje karakteristicnih tacaka
    # predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    #
    # for image_path in train_image_paths:
    #     image = cv2.imread(image_path)
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #
    #     rects = detector(gray, 1)
    #
    #     # iteriramo kroz sve detekcije korak 1.
    #     for (i, rect) in enumerate(rects):
    #         # determine the facial landmarks for the face region, then
    #         # convert the facial landmark (x, y)-coordinates to a NumPy
    #         # array
    #         # odredjivanje kljucnih tacaka - korak 2
    #         shape = predictor(gray, rect)
    #         # shape predstavlja 68 koordinata
    #         shape = face_utils.shape_to_np(shape)  # konverzija u NumPy niz
    #         print("Dimenzije prediktor matrice: {0}".format(shape.shape))  # 68 tacaka (x,y)
    #         print("Prva 3 elementa matrice")
    #         print(shape[:3])
    #
    #         # konvertovanje pravougaonika u bounding box koorinate
    #         (x, y, w, h) = face_utils.rect_to_bb(rect)
    #         # crtanje pravougaonika oko detektovanog lica
    #         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #
    #         # ispis rednog broja detektovanog lica
    #         cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
    #         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    #
    #         # crtanje kljucnih tacaka
    #         for (x, y) in shape:
    #             cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
    #
    #         plt.imshow(image)
    #         plt.show()


    model = generate_model(train_image_paths, train_image_labels)
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

    model = generate_model(train_image_paths, train_image_labels)
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

    model = generate_model(train_image_paths, train_image_labels)
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
    img = cv2.resize(load_image(image_path), (200,200), interpolation=cv2.INTER_NEAREST)
    hog = create_hog_descriptor(img.shape)
    age = trained_model.predict(reshape_data(np.array([hog.compute(img)])))
    return age[0]

def predict_gender(trained_model, image_path):
    """
    Procedura prima objekat istreniranog modela za prepoznavanje pola na osnovu lica i putanju do fotografije na kojoj
    se nalazi novo lice sa koga treba prepoznati pol.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param trained_model: <Model> Istrenirani model za prepoznavanje karaktera
    :param image_path: <String> Putanja do fotografije sa koje treba prepoznati ekspresiju lica
    :return: <Int>  Prepoznata klasa pola (0 - musko, 1 - zensko)
    """
    gender = 1
    img = cv2.resize(load_image(image_path), (200,200), interpolation=cv2.INTER_NEAREST)
    hog = create_hog_descriptor(img.shape)
    gender = trained_model.predict(reshape_data(np.array([hog.compute(img)])))

    return gender[0]
def predict_race(trained_model, image_path):
    """
    Procedura prima objekat istreniranog modela za prepoznavanje rase lica i putanju do fotografije na kojoj
    se nalazi novo lice sa koga treba prepoznati rasu.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param trained_model: <Model> Istrenirani model za prepoznavanje karaktera
    :param image_path: <String> Putanja do fotografije sa koje treba prepoznati ekspresiju lica
    :return: <Int>  Prepoznata klasa (0 - Bela, 1 - Crna, 2 - Azijati, 3- Indijci, 4 - Ostali)
    """
    race = 1
    img = cv2.resize(load_image(image_path), (200, 200), interpolation=cv2.INTER_NEAREST)
    hog = create_hog_descriptor(img.shape)
    race = trained_model.predict(reshape_data(np.array([hog.compute(img)])))

    return race[0]
