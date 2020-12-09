# import libraries here
import cv2
import numpy as np
import matplotlib.pyplot as plt

# keras
# from keras.models import Sequential
# from keras.layers.core import Dense, Activation
# from keras.optimizers import SGD
# from keras.models import model_from_json
# from keras.models import load_model

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

    # inicijalizaclija dlib detektora (HOG)
    detector = dlib.get_frontal_face_detector()
    # ucitavanje pretreniranog modela za prepoznavanje karakteristicnih tacaka
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


    model = None
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

    model = None
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

    model = None
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
    age = np.random.randint(3,72)

    return age

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
    gender = np.random.randint(0, 2)

    return gender

def predict_race(trained_model, image_path):
    """
    Procedura prima objekat istreniranog modela za prepoznavanje rase lica i putanju do fotografije na kojoj
    se nalazi novo lice sa koga treba prepoznati rasu.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param trained_model: <Model> Istrenirani model za prepoznavanje karaktera
    :param image_path: <String> Putanja do fotografije sa koje treba prepoznati ekspresiju lica
    :return: <Int>  Prepoznata klasa (0 - Bela, 1 - Crna, 2 - Azijati, 3- Indijci, 4 - Ostali)
    """
    race = 4
    race = np.random.randint(0, 5)

    return race 
