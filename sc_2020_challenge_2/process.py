# import libraries here
from __future__ import print_function

import cv2
import numpy as np
import matplotlib.pyplot as plt

from imutils import face_utils
import argparse
import imutils
import dlib

# keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.models import model_from_json
from keras.models import load_model

from fuzzywuzzy import fuzz
from fuzzywuzzy import process

# Sklearn biblioteka sa implementiranim K-means algoritmom
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

import matplotlib.pylab as pylab




def load_image(path):

    # inicijalizaclija dlib detektora (HOG)
    detector = dlib.get_frontal_face_detector()
    # ucitavanje pretreniranog modela za prepoznavanje karakteristicnih tacaka
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret, image_bin = cv2.threshold(image_gs, 207, 255, cv2.THRESH_BINARY)
    display_image(image_bin)

    # blur = cv2.GaussianBlur(image_gs, (5, 5), 0)
    # display_image(blur)
    image_bin_adaptive = cv2.threshold(image_gs, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    display_image(image_bin_adaptive)

    return image_bin


def invert(image):
    return 255 - image


def display_image(image, color=False):
    if color:
        plt.imshow(image)
        plt.show()
    else:
        plt.imshow(image, 'gray')
        plt.show()


def dilate(image):
    kernel = np.ones((2, 2))  # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)


def erode(image):
    kernel = np.ones((2, 2))  # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)


def resize_region(region):
    '''Transformisati selektovani region na sliku dimenzija 28x28'''

    if region.shape[0] == 0 or region.shape[1] == 0:
        return None
    return cv2.resize(region, (100, 100), interpolation=cv2.INTER_NEAREST)


def isAlreadyAdded(regions_array_filtered, x):
    for region in regions_array_filtered:
        if region[1][0] == x:
            return True

    return False


def select_roi(image_orig, image_bin):
    '''Oznaciti regione od interesa na originalnoj slici. (ROI = regions of interest)
        Za svaki region napraviti posebnu sliku dimenzija 28 x 28.
        Za označavanje regiona koristiti metodu cv2.boundingRect(contour).
        Kao povratnu vrednost vratiti originalnu sliku na kojoj su obeleženi regioni
        i niz slika koje predstavljaju regione sortirane po rastućoj vrednosti x ose
    '''
    # if cancel:
    #     return None
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = []  # lista sortiranih regiona po x osi (sa leva na desno)
    regions_array = []
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])  # koordinate i velicina granicnog pravougaonika
        area = cv2.contourArea(contours[i])
        if h > 10 and w > 10 and hierarchy[0, i, 3] == -1:
            # kopirati [y:y+h+1, x:x+w+1] sa binarne slike i smestiti u novu sliku
            # označiti region pravougaonikom na originalnoj slici (image_orig) sa rectangle funkcijom
            region = image_bin[y:y + h + 1, x:x + w + 1]
            # display_image(region)
            regions_array.append([resize_region(region), (x, y, w, h)])
            # cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # sortirati sve regione po x osi (sa leva na desno) i smestiti u promenljivu sorted_regions
    regions_array = sorted(regions_array, key=lambda item: item[1][0])

    regions_array_filtered = []

    for region in regions_array:
        x1 = region[1][0]
        y1 = region[1][1]
        w1 = region[1][2]
        h1 = region[1][3]
        found = False
        for smaller_region in regions_array:
            x2 = smaller_region[1][0]
            y2 = smaller_region[1][1]
            w2 = smaller_region[1][2]
            h2 = smaller_region[1][3]
            if x2 > x1 and x2 + w2 < x1 + w1:
                found = True
                x = x1
                y = y2 - 10
                w = w1
                h = h1 + h2 + 20
                if not isAlreadyAdded(regions_array_filtered, x):
                    cutout = image_bin[y:y + h + 1, x:x + w + 1]
                    regions_array_filtered.append([resize_region(cutout), (x, y, w, h)])
                    cv2.rectangle(image_orig, (x, y), (x + w, y + h), (255, 0, 0), 2)
            elif x2 < x1 and x2 + w2 > x1 + w1:
                found = True
                x = x2
                y = y1 - 10
                w = w2
                h = h1 + h2 + 20
                if not isAlreadyAdded(regions_array_filtered, x):
                    cutout = image_bin[y:y + h + 1, x:x + w + 1]
                    regions_array_filtered.append([resize_region(cutout), (x, y, w, h)])
                    cv2.rectangle(image_orig, (x, y), (x + w, y + h), (255, 0, 0), 2)
        if not found:
            cutout = image_bin[y1:y1 + h1 + 1, x1:x1 + w1 + 1]
            regions_array_filtered.append([resize_region(cutout), (x1, y1, w1, h1)])
            cv2.rectangle(image_orig, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)

    sorted_regions = [region[0] for region in regions_array_filtered]

    sorted_rectangles = [region[1] for region in regions_array_filtered]
    region_distances = []
    # Izdvojiti sortirane parametre opisujućih pravougaonika
    # Izračunati rastojanja između svih susednih regiona po x osi i dodati ih u region_distances niz
    for index in range(0, len(sorted_rectangles) - 1):
        current = sorted_rectangles[index]
        next_rect = sorted_rectangles[index + 1]
        distance = next_rect[0] - (current[0] + current[2])  # X_next - (X_current + W_current)
        region_distances.append(distance)

    # computer countour center
    centers = []
    for contour in sorted_rectangles:  # x, y, w, h
        cx = contour[0] + contour[2] / 2
        cy = contour[1] + contour[3] / 2
        centers.append((cx, cy))

    return image_orig, sorted_regions, region_distances, centers


def select_roi_train(image_orig, image_bin):
    '''Oznaciti regione od interesa na originalnoj slici. (ROI = regions of interest)
        Za svaki region napraviti posebnu sliku dimenzija 28 x 28.
        Za označavanje regiona koristiti metodu cv2.boundingRect(contour).
        Kao povratnu vrednost vratiti originalnu sliku na kojoj su obeleženi regioni
        i niz slika koje predstavljaju regione sortirane po rastućoj vrednosti x ose
    '''
    # if cancel:
    #     return None
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = []  # lista sortiranih regiona po x osi (sa leva na desno)
    regions_array = []
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])  # koordinate i velicina granicnog pravougaonika
        area = cv2.contourArea(contours[i])
        if w > 5 and h > 10 and hierarchy[0, i, 3] == -1:
            region = image_bin[y:y + h + 1, x:x + w + 1]
            # display_image(region)
            regions_array.append([resize_region(region), (x, y, w, h)])
            # cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # sortirati sve regione po x osi (sa leva na desno) i smestiti u promenljivu sorted_regions
    regions_array = sorted(regions_array, key=lambda item: item[1][0])

    regions_array_filtered = []

    for region in regions_array:
        x1 = region[1][0]
        y1 = region[1][1]
        w1 = region[1][2]
        h1 = region[1][3]
        found = False
        for smaller_region in regions_array:
            x2 = smaller_region[1][0]
            y2 = smaller_region[1][1]
            w2 = smaller_region[1][2]
            h2 = smaller_region[1][3]
            if x2 > x1 and x2 + w2 < x1 + w1:
                found = True
                x = x1
                y = y2
                w = w1
                h = h1 + h2
                if not isAlreadyAdded(regions_array_filtered, x):
                    cutout = image_bin[y:y + h + 1, x:x + w + 1]
                    regions_array_filtered.append([resize_region(cutout), (x, y, w, h)])
                    cv2.rectangle(image_orig, (x, y), (x + w, y + h), (255, 0, 0), 2)
            elif x2 < x1 and x2 + w2 > x1 + w1:
                found = True
                x = x2
                y = y1
                w = w2
                h = h1 + h2
                if not isAlreadyAdded(regions_array_filtered, x):
                    cutout = image_bin[y:y + h + 1, x:x + w + 1]
                    regions_array_filtered.append([resize_region(cutout), (x, y, w, h)])
                    cv2.rectangle(image_orig, (x, y), (x + w, y + h), (255, 0, 0), 2)
        if not found:
            cutout = image_bin[y1:y1 + h1 + 1, x1:x1 + w1 + 1]
            regions_array_filtered.append([resize_region(cutout), (x1, y1, w1, h1)])
            cv2.rectangle(image_orig, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)

    sorted_regions = [region[0] for region in regions_array_filtered]

    sorted_rectangles = [region[1] for region in regions_array_filtered]
    region_distances = []
    # Izdvojiti sortirane parametre opisujućih pravougaonika
    # Izračunati rastojanja između svih susednih regiona po x osi i dodati ih u region_distances niz
    for index in range(0, len(sorted_rectangles) - 1):
        current = sorted_rectangles[index]
        next_rect = sorted_rectangles[index + 1]
        distance = next_rect[0] - (current[0] + current[2])  # X_next - (X_current + W_current)
        region_distances.append(distance)

    # computer countour center
    centers = []
    for contour in sorted_rectangles:  # x, y, w, h
        cx = contour[0] + contour[2] / 2
        cy = contour[1] + contour[3] / 2
        centers.append((cx, cy))

    return image_orig, sorted_regions, region_distances, centers


def select_roi_train_paper(image_orig, image_bin):
    '''Oznaciti regione od interesa na originalnoj slici. (ROI = regions of interest)
        Za svaki region napraviti posebnu sliku dimenzija 28 x 28.
        Za označavanje regiona koristiti metodu cv2.boundingRect(contour).
        Kao povratnu vrednost vratiti originalnu sliku na kojoj su obeleženi regioni
        i niz slika koje predstavljaju regione sortirane po rastućoj vrednosti x ose
    '''
    # if cancel:
    #     return None

    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = []
    regions_array = []
    for i in range(0, len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])  # koordinate i velicina granicnog pravougaonika
        area = cv2.contourArea(contours[i])
        #if w > 5 and h > 10:
        region = image_bin[y:y + h + 1, x:x + w + 1]
        # display_image(region)
        regions_array.append([resize_region(contours[i]), (x, y, w, h)])
        cv2.rectangle(image_bin, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # sortirati sve regione po x osi (sa leva na desno) i smestiti u promenljivu sorted_regions
    display_image(image_bin)
    regions_array = sorted(regions_array, key=lambda item: item[1][0])

    regions_array_filtered = []
    for region in regions_array:
        x1 = region[1][0]
        y1 = region[1][1]
        w1 = region[1][2]
        h1 = region[1][3]
        found = False
        for smaller_region in regions_array:
            x2 = smaller_region[1][0]
            y2 = smaller_region[1][1]
            w2 = smaller_region[1][2]
            h2 = smaller_region[1][3]
            if x2 > x1 and x2 + w2 < x1 + w1:
                found = True
                x = x1
                y = y2
                w = w1
                h = h1 + h2
                if not isAlreadyAdded(regions_array_filtered, x):
                    cutout = image_bin[y:y + h + 1, x:x + w + 1]
                    regions_array_filtered.append([resize_region(cutout), (x, y, w, h)])
                    cv2.rectangle(image_orig, (x, y), (x + w, y + h), (255, 0, 0), 2)
            elif x2 < x1 and x2 + w2 > x1 + w1:
                found = True
                x = x2
                y = y1
                w = w2
                h = h1 + h2
                if not isAlreadyAdded(regions_array_filtered, x):
                    cutout = image_bin[y:y + h + 1, x:x + w + 1]
                    regions_array_filtered.append([resize_region(cutout), (x, y, w, h)])
                    cv2.rectangle(image_orig, (x, y), (x + w, y + h), (255, 0, 0), 2)
        if not found:
            cutout = image_bin[y1:y1 + h1 + 1, x1:x1 + w1 + 1]
            regions_array_filtered.append([resize_region(cutout), (x1, y1, w1, h1)])
            cv2.rectangle(image_orig, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)

    sorted_regions = [region[0] for region in regions_array_filtered]

    sorted_rectangles = [region[1] for region in regions_array_filtered]
    region_distances = []
    # Izdvojiti sortirane parametre opisujućih pravougaonika
    # Izračunati rastojanja između svih susednih regiona po x osi i dodati ih u region_distances niz
    for index in range(0, len(sorted_rectangles) - 1):
        current = sorted_rectangles[index]
        next_rect = sorted_rectangles[index + 1]
        distance = next_rect[0] - (current[0] + current[2])  # X_next - (X_current + W_current)
        region_distances.append(distance)

    # computer countour center
    centers = []
    for contour in sorted_rectangles:  # x, y, w, h
        cx = contour[0] + contour[2] / 2
        cy = contour[1] + contour[3] / 2
        centers.append((cx, cy))

    return image_orig, sorted_regions, region_distances, centers

def select_roi_train_final_paper(image_orig, image_bin):
    '''Oznaciti regione od interesa na originalnoj slici. (ROI = regions of interest)
        Za svaki region napraviti posebnu sliku dimenzija 28 x 28.
        Za označavanje regiona koristiti metodu cv2.boundingRect(contour).
        Kao povratnu vrednost vratiti originalnu sliku na kojoj su obeleženi regioni
        i niz slika koje predstavljaju regione sortirane po rastućoj vrednosti x ose
    '''
    # if cancel:
    #     return None
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = []  # lista sortiranih regiona po x osi (sa leva na desno)
    dummy = cv2.cvtColor(image_bin.copy(), cv2.COLOR_GRAY2RGB)
    regions_array = []
    for i in range(1, len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])  # koordinate i velicina granicnog pravougaonika
        area = cv2.contourArea(contours[i])
        if w > 5 and h > 20 and  hierarchy[0, i, 3] == -1 and w<100:
            region = image_bin[y:y + h + 1, x:x + w + 1]
            # display_image(region)
            regions_array.append([resize_region(region), (x, y, w, h)])
            cv2.rectangle(dummy, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # sortirati sve regione po x osi (sa leva na desno) i smestiti u promenljivu sorted_regions
    display_image(dummy)
    regions_array = sorted(regions_array, key=lambda item: item[1][0])


    sorted_regions = [region[0] for region in regions_array]
    sorted_rectangles = [region[1] for region in regions_array]

    region_distances = []

    # Izdvojiti sortirane parametre opisujućih pravougaonika
    # Izračunati rastojanja između svih susednih regiona po x osi i dodati ih u region_distances niz
    for index in range(0, len(sorted_rectangles) - 1):
        current = sorted_rectangles[index]
        next_rect = sorted_rectangles[index + 1]
        distance = next_rect[0] - (current[0] + current[2])  # X_next - (X_current + W_current)
        region_distances.append(distance)

    # computer countour center
    centers = []
    for contour in sorted_rectangles:  # x, y, w, h
        cx = contour[0] + contour[2] / 2
        cy = contour[1] + contour[3] / 2
        centers.append((cx, cy))

    return image_bin, sorted_regions, region_distances, centers

def scale_to_range(image):  # skalira elemente slike na opseg od 0 do 1
    ''' Elementi matrice image su vrednosti 0 ili 255.
        Potrebno je skalirati sve elemente matrica na opseg od 0 do 1
    '''
    return image / 255


def matrix_to_vector(image):
    '''Sliku koja je zapravo matrica 28x28 transformisati u vektor sa 784 elementa'''
    return image.flatten()


def prepare_for_ann(regions):
    '''Regioni su matrice dimenzija 28x28 čiji su elementi vrednosti 0 ili 255.
        Potrebno je skalirati elemente regiona na [0,1] i transformisati ga u vektor od 784 elementa '''
    ready_for_ann = []
    for region in regions:
        # skalirati elemente regiona
        # region sa skaliranim elementima pretvoriti u vektor
        # vektor dodati u listu spremnih regiona
        scale = scale_to_range(region)
        ready_for_ann.append(matrix_to_vector(scale))

    return ready_for_ann


def convert_output(alphabet):
    '''Konvertovati alfabet u niz pogodan za obučavanje NM,
        odnosno niz čiji su svi elementi 0 osim elementa čiji je
        indeks jednak indeksu elementa iz alfabeta za koji formiramo niz.
        Primer prvi element iz alfabeta [1,0,0,0,0,0,0,0,0,0],
        za drugi [0,1,0,0,0,0,0,0,0,0] itd..
    '''
    nn_outputs = []
    for index in range(len(alphabet)):
        output = np.zeros(len(alphabet))
        output[index] = 1
        nn_outputs.append(output)
    return np.array(nn_outputs)


def create_ann():
    '''Implementacija veštačke neuronske mreže sa 784 neurona na uloznom sloju,
        128 neurona u skrivenom sloju i 10 neurona na izlazu. Aktivaciona funkcija je sigmoid.
    '''
    ann = Sequential()
    ann.add(Dense(256, input_dim=10000, activation='sigmoid'))
    ann.add(Dense(60, activation='sigmoid'))
    return ann


def train_ann(ann, X_train, y_train):
    '''Obucavanje vestacke neuronske mreze'''
    X_train = np.array(X_train, np.float32)  # dati ulazi
    y_train = np.array(y_train, np.float32)  # zeljeni izlazi za date ulaze

    # definisanje parametra algoritma za obucavanje
    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)

    # obucavanje neuronske mreze
    ann.fit(X_train, y_train, epochs=1500, batch_size=1, verbose=0, shuffle=True)

    return ann


def winner(output):  # output je vektor sa izlaza neuronske mreze
    """pronaći i vratiti indeks neurona koji je najviše pobuđen"""
    return max(enumerate(output), key=lambda x: x[1])[0]


def display_result(outputs, alphabet):
    '''za svaki rezultat pronaći indeks pobedničkog
        regiona koji ujedno predstavlja i indeks u alfabetu.
        Dodati karakter iz alfabet u result'''
    result = []
    for output in outputs:
        result.append(alphabet[winner(output)])
    return result


def display_result_with_distances(outputs, alphabet, k_means):
    '''
    Funkcija određuje koja od grupa predstavlja razmak između reči, a koja između slova, i na osnovu
    toga formira string od elemenata pronađenih sa slike.
    Args:
        outputs: niz izlaza iz neuronske mreže.
        alphabet: niz karaktera koje je potrebno prepoznati
        kmeans: obučen kmeans objekat
    Return:
        Vraća formatiran string
    '''
    # Odrediti indeks grupe koja odgovara rastojanju između reči, pomoću vrednosti iz k_means.cluster_centers_
    w_space_group = max(enumerate(k_means.cluster_centers_), key=lambda x: x[1])[0]
    result = alphabet[winner(outputs[0])]
    for idx, output in enumerate(outputs[1:, :]):
        # Iterativno dodavati prepoznate elemente kao u vežbi 2, alphabet[winner(output)]
        # Dodati space karakter u slučaju da odgovarajuće rastojanje između dva slova odgovara razmaku između reči.
        # U ovu svrhu, koristiti atribut niz k_means.labels_ koji sadrži sortirana rastojanja između susednih slova.
        if (k_means.labels_[idx] == w_space_group):
            result += ' '
        result += alphabet[winner(output)]
    return result


def serialize_ann(ann):
    # serijalizuj arhitekturu neuronske mreze u JSON fajl
    model_json = ann.to_json()
    with open("serialized_model/neuronska.json", "w") as json_file:
        json_file.write(model_json)
    # serijalizuj tezine u HDF5 fajl
    ann.save_weights("serialized_model/neuronska.h5")
    # ann.save("my_model")


def load_trained_ann():
    try:
        # Ucitaj JSON i kreiraj arhitekturu neuronske mreze na osnovu njega
        json_file = open('serialized_model/neuronska.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        ann = model_from_json(loaded_model_json)
        # ucitaj tezine u prethodno kreirani model
        ann.load_weights("serialized_model/neuronska.h5")
        print("Istrenirani model uspesno ucitan.")
        # ann = tf.keras.models.load_model("my_model", compile=False)
        return ann
    except Exception as e:
        print(e)
        # ako ucitavanje nije uspelo, verovatno model prethodno nije serijalizovan pa nema odakle da bude ucitan
        return None


def create_inputs(train_image_paths):
    inputs = []
    for i in range(len(train_image_paths)):
        image_color = load_image(train_image_paths[i])
        display_image(image_color)
        img = invert(image_bin(image_gray(image_color)))
        display_image(img)
        img_bin = erode(dilate(img))
        display_image(img_bin)
        selected_regions, letters, distances, centers = select_roi(image_color.copy(), img)
        display_image(selected_regions)
        for result in prepare_for_ann(letters):
            inputs.append(result)

    return inputs


def train_or_load_character_recognition_model(train_image_paths, serialization_folder):
    """
    Procedura prima putanje do fotografija za obucavanje (dataset se sastoji iz razlicitih fotografija alfabeta), kao i
    putanju do foldera u koji treba sacuvati model nakon sto se istrenira (da ne trenirate svaki put iznova)

    Procedura treba da istrenira model i da ga sacuva u folder "serialization_folder" pod proizvoljnim nazivom

    Kada se procedura pozove, ona treba da trenira model ako on nije istraniran, ili da ga samo ucita ako je prethodno
    istreniran i ako se nalazi u folderu za serijalizaciju

    :param train_image_paths: putanje do fotografija alfabeta
    :param serialization_folder: folder u koji treba sacuvati serijalizovani model
    :return: Objekat modela
    """
    # TODO - Istrenirati model ako vec nije istreniran, ili ga samo ucitati iz foldera za serijalizaciju

    alphabet = ['A', 'B', 'C', 'Č', 'Ć', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
                'R', 'S', 'Š', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Ž', 'a', 'b', 'c', 'č', 'ć', 'd', 'e',
                'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
                'r', 's', 'š', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ž']

    # probaj da ucitas prethodno istreniran model
    ann = load_trained_ann()

    # ako je ann=None, znaci da model nije ucitan u prethodnoj metodi i da je potrebno istrenirati novu mrezu
    if ann == None:
        print("Traniranje modela zapoceto.")
        ann = create_ann()
        inputs = create_inputs(train_image_paths)
        outputs = convert_output(alphabet)
        ann = train_ann(ann, inputs, outputs)
        print("Treniranje modela zavrseno.")
        # serijalizuj novu mrezu nakon treniranja, da se ne trenira ponovo svaki put
        serialize_ann(ann)

    return ann


def postprocess(extracted_text, vocabulary):
    vocabulary_words = vocabulary.keys()

    text_splits = extracted_text.split(" ")
    improved_text = ""
    i = 0
    for word in text_splits:
        highest = process.extractOne(word, vocabulary_words)
        temp = highest[0]
        if i > 0 and highest[0] != 'I':
            temp = str.lower(highest[0])
        improved_text = improved_text + temp + " "
        i = i + 1

    ret = improved_text[:-1]

    return ret


def fix_t(extracted_text):
    text_splits = extracted_text.split(" ")
    improved_text = ""
    i = 0
    for word in text_splits:
        temp = word
        if word == 'T':
            temp = 'I'
        improved_text = improved_text + temp + " "
        i = i + 1

    ret = improved_text[:-1]
    return ret


def make_text_lowercase(extracted_text):
    text_splits = extracted_text.split(" ")
    improved_text = ""
    i = 0
    for word in text_splits:
        temp = word
        if i > 0 and word != 'I':
            temp = str.lower(word)
        improved_text = improved_text + temp + " "
        i = i + 1

    ret = improved_text[:-1]
    return ret


def postprocess_levenstein(extracted_text, vocabulary):
    vocabulary_words = vocabulary.keys()

    extracted_text = fix_t(extracted_text)
    extracted_text = make_text_lowercase(extracted_text)

    text_splits = extracted_text.split(" ")
    improved_text = ""
    i = 0
    for word in text_splits:
        distances = []
        for key in vocabulary_words:
            ratio = fuzz.ratio(word, key)
            distances.append((key, ratio))
        distances = sorted(distances, key=lambda item: item[1], reverse=True)
        highest = max(distances, key=lambda item: item[1])
        highest_levenstein = highest[1]

        final_list = []
        for j in range(0, 10):
            if distances[j][1] == highest_levenstein:
                final_list.append(distances[j][0])

        final_word = ""
        max_appearance = -1
        if len(final_list) != 0:
            for word in final_list:
                if int(vocabulary[word]) > max_appearance:
                    final_word = word
                    max_appearance = int(vocabulary[word])
        improved_text = improved_text + final_word + " "
        i = i + 1

    ret = improved_text[:-1]

    return ret


def get_angle(k):
    radians = np.arctan(k)
    degrees = np.rad2deg(radians)
    return degrees


def is_error(extracted_text):
    text_splits = extracted_text.split(" ")
    one_word = 0
    all_words = len(text_splits)
    for word in text_splits:
        if len(word) == 1:
            one_word = one_word + 1

    if one_word / all_words > 0.5:
        return True
    return False


def extract_text_from_image(trained_model, image_path, vocabulary):
    """
    Procedura prima objekat istreniranog modela za prepoznavanje znakova (karaktera), putanju do fotografije na kojoj
    se nalazi tekst za ekstrakciju i recnik svih poznatih reci koje se mogu naci na fotografiji.
    Procedura treba da ucita fotografiju sa prosledjene putanje, i da sa nje izvuce sav tekst koriscenjem
    openCV (detekcija karaktera) i prethodno istreniranog modela (prepoznavanje karaktera), i da vrati procitani tekst
    kao string.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param trained_model: <Model> Istrenirani model za prepoznavanje karaktera
    :param image_path: <String> Putanja do fotografije sa koje treba procitati tekst.
    :param vocabulary: <Dict> Recnik SVIH poznatih reci i ucestalost njihovog pojavljivanja u tekstu
    :return: <String>  Tekst procitan sa ulazne slike
    """
    try:
        extracted_text = ""
        # TODO - Izvuci tekst sa ulazne fotografije i vratiti ga kao string

        alphabet = ['A', 'B', 'C', 'Č', 'Ć', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
                    'R', 'S', 'Š', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Ž', 'a', 'b', 'c', 'č', 'ć', 'd', 'e',
                    'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
                    'r', 's', 'š', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ž']

        inputs = []
        image_color = load_image(image_path)
        # display_image(image_color)
        img = invert(image_bin(image_gray(image_color)))
        # display_image(img)
        img_bin = erode(dilate(img))
        # display_image(img_bin)

        # if image_path =='.\\dataset\\validation\\train84.png':
        #     a = 4

        selected_regions, letters, distances, centers = select_roi_train(image_color.copy(), img)

        p0, p1, p2, p3 = cv2.fitLine(np.array(centers), cv2.DIST_L1, 0, 0.1, 0.01)
        temp = cv2.imread(image_path)
        height, width, channels = temp.shape

        x0 = p2
        y0 = p3
        k = p1 / p0

        angle_in_degrees = get_angle(k)

        point1X = 0
        point1Y = k * (0 - x0) + y0
        point2X = width
        point2Y = k * (width - x0) + y0
        # if point1Y > height or point2Y > height:
        #     raise Exception
        # cv2.line(image_color, (point1X, point1Y), (point2X,point2Y), (0, 255, 0), thickness=15)
        # display_image(image_color)



        (h, w) = image_color.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle_in_degrees, 1.0)
        newImage = cv2.warpAffine(image_color, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        #display_image(newImage)

        img = invert(image_bin(image_gray(newImage)))

        selected_regions, letters, distances, centers = select_roi_train(newImage.copy(), img)
        #display_image(selected_regions)

        # cv2.imshow()
        # waitkey
        # cv2.imwrite()

        for result in prepare_for_ann(letters):
            inputs.append(result)

        # Podešavanje centara grupa K-means algoritmom
        distances = np.array(distances).reshape(len(distances), 1)
        # Neophodno je da u K-means algoritam bude prosleđena matrica u kojoj vrste određuju elemente
        k_means = KMeans(n_clusters=2, max_iter=2000, tol=0.00001, n_init=10)
        k_means.fit(distances)

        # inputs = prepare_for_ann(letters)
        results = trained_model.predict(np.array(inputs, np.float32))

        extracted_text = display_result_with_distances(results, alphabet, k_means)
        print('\n' + image_path)
        print("extracted : " + extracted_text)

        improved_text = postprocess_levenstein(extracted_text, vocabulary)
        print("fuzzy improved : " + improved_text)

        return improved_text

    except Exception as e:
        try:
            # todo blue background
            print(e)

            image_color = load_image(image_path)

            alphabet = ['A', 'B', 'C', 'Č', 'Ć', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
                        'R', 'S', 'Š', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Ž', 'a', 'b', 'c', 'č', 'ć', 'd', 'e',
                        'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
                        'r', 's', 'š', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ž']

            r = image_color.copy()
            # set blue and green channels to 0
            r[:, :, 0] = 0
            r[:, :, 1] = 0

            # RGB - Red
            plt.imshow(r)
            plt.show()

            # img = invert(image_bin(image_gray(r)))
            image_bin_adaptive = cv2.threshold(image_gray(r), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            #image_bin_adaptive = cv2.threshold(image_gray(r), 0, 215, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)[1]
            # display_image(image_bin_adaptive)
            inverted_image = invert(image_bin_adaptive)

            image_color = r.copy()
            inputs = []

            selected_regions, letters, distances, centers = select_roi_train(image_color.copy(), inverted_image)

            p0, p1, p2, p3 = cv2.fitLine(np.array(centers), cv2.DIST_L1, 0, 0.1, 0.01)
            temp = cv2.imread(image_path)
            height, width, channels = temp.shape

            k = p1 / p0

            angle_in_degrees = get_angle(k)

            (h, w) = image_color.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle_in_degrees, 1.0)
            newImage = cv2.warpAffine(image_color, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            #display_image(newImage)

            image_bin_adaptive = cv2.threshold(image_gray(newImage), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            inverted_image = invert(image_bin_adaptive)
            #display_image(inverted_image)

            selected_regions, letters, distances, centers = select_roi_train(newImage.copy(), inverted_image)
            #display_image(selected_regions)

            for result in prepare_for_ann(letters):
                inputs.append(result)

            distances = np.array(distances).reshape(len(distances), 1)
            k_means = KMeans(n_clusters=2, max_iter=2000, tol=0.00001, n_init=10)
            k_means.fit(distances)

            results = trained_model.predict(np.array(inputs, np.float32))

            extracted_text = display_result_with_distances(results, alphabet, k_means)
            print('\n' + image_path)
            print("extracted : " + extracted_text)

            if is_error(extracted_text):
                return ""

            improved_text = postprocess_levenstein(extracted_text, vocabulary)
            print("fuzzy improved : " + improved_text)

            return improved_text
        # light_gray = (229, 193, 248)
        # dark_black = (214, 181, 232)
        except Exception as e:
            try:
                # todo paper background
                image_color = load_image(image_path)
                display_image(image_color)
                img = invert(image_bin(image_gray(image_color)))
                #display_image(img)
                img_bin = erode(img)
                img_bin = invert(img_bin)
                display_image(img_bin)


                alphabet = ['A', 'B', 'C', 'Č', 'Ć', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
                            'Q',
                            'R', 'S', 'Š', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Ž', 'a', 'b', 'c', 'č', 'ć', 'd', 'e',
                            'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
                            'r', 's', 'š', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ž']

                inputs = []
                img = cv2.copyMakeBorder(img_bin, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                image_color = cv2.copyMakeBorder(image_color, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                display_image(img)

                selected_regions, letters, distances, centers = select_roi_train_final_paper(image_color.copy(), img.copy())
                display_image(selected_regions)

                try:
                    p0, p1, p2, p3 = cv2.fitLine(np.array(centers), cv2.DIST_L1, 0, 0.1, 0.01)
                except Exception as e:
                    return ""
                temp = cv2.imread(image_path)
                height, width, channels = temp.shape
                k = p1 / p0
                angle_in_degrees = get_angle(k)

                (h, w) = image_color.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle_in_degrees, 1.0)
                newImage = cv2.warpAffine(image_color, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                display_image(newImage)

                img = invert(image_bin(image_gray(newImage)))
                img_bin = erode(img)
                img_bin = invert(img_bin)
                display_image(img_bin)

                img = cv2.copyMakeBorder(img_bin, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                newImage = cv2.copyMakeBorder(newImage, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(255, 255, 255))

                selected_regions, letters, distances, centers = select_roi_train_final_paper(newImage.copy(), img)
                display_image(selected_regions)

                for result in prepare_for_ann(letters):
                    inputs.append(result)

                distances = np.array(distances).reshape(len(distances), 1)
                k_means = KMeans(n_clusters=2, max_iter=2000, tol=0.00001, n_init=10)
                k_means.fit(distances)

                results = trained_model.predict(np.array(inputs, np.float32))

                extracted_text = display_result_with_distances(results, alphabet, k_means)
                print('\n' + image_path)
                print("extracted : " + extracted_text)

                if is_error(extracted_text):
                    raise Exception

                improved_text = postprocess_levenstein(extracted_text, vocabulary)
                print("fuzzy improved : " + improved_text)

                return improved_text
            except Exception as e:
                # print(image_path + "  " + e)
                return ""


# pokusaj i da sredis
# i,l jako tesko detektuje
# tweakuj neuronsku
# pozadinske slike da resis
