# import libraries here
from __future__ import print_function
import cv2
import numpy as np
import matplotlib.pyplot as plt
import collections

import math

# keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.models import model_from_json
from keras.models import load_model

# Sklearn biblioteka sa implementiranim K-means algoritmom
from sklearn import datasets
from sklearn.cluster import KMeans

import matplotlib.pylab as pylab
cancel = False

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret, image_bin = cv2.threshold(image_gs, 200, 255, cv2.THRESH_BINARY)
    display_image(image_bin)
    # image_bin = cv2.adaptiveThreshold(image_gs, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
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
    kernel = np.ones((3, 3))  # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)


def erode(image):
    kernel = np.ones((3, 3))  # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)


def resize_region(region):
    '''Transformisati selektovani region na sliku dimenzija 28x28'''
    global cancel
    if region.shape[0] == 0 or region.shape[1]==0:
        cancel = True
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
            # elif x1 < x2 < x1+w1:
            #     x = x1
            #     y= y2-10
            #     w = 2*w2 - w1
            #     h = h1 + h2+20
            #     if not isAlreadyAdded(regions_array_filtered, x):
            #         cutout = image_bin[y:y + h + 1, x:x + w + 1]
            #         regions_array_filtered.append([resize_region(cutout), (x, y, w, h)])
            #         cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 0, 0), 2)
            # elif x2 < x1 <x2+w2:
            #     x = x2
            #     y= y1-10
            #     w = 2*w1 - w2
            #     h = h1 + h2 + 20
            #     if not isAlreadyAdded(regions_array_filtered, x):
            #         cutout = image_bin[y:y + h + 1, x:x + w + 1]
            #         regions_array_filtered.append([resize_region(cutout), (x, y, w, h)])
            #         cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 0, 0), 2)
        if not found and w1 > 20:
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

    return image_orig, sorted_regions, region_distances


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
    ann.add(Dense(128, input_dim=10000, activation='sigmoid'))
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
    ann.fit(X_train, y_train, epochs=500, batch_size=1, verbose=0, shuffle=False)

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
        selected_regions, letters, distances = select_roi(image_color.copy(), img)
        # display_image(numbers[0])
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
    inputs = create_inputs(train_image_paths)

    # probaj da ucitas prethodno istreniran model
    ann = load_trained_ann()

    # ako je ann=None, znaci da model nije ucitan u prethodnoj metodi i da je potrebno istrenirati novu mrezu
    if ann == None:
        print("Traniranje modela zapoceto.")
        ann = create_ann()
        outputs = convert_output(alphabet)
        ann = train_ann(ann, inputs, outputs)
        print("Treniranje modela zavrseno.")
        # serijalizuj novu mrezu nakon treniranja, da se ne trenira ponovo svaki put
        serialize_ann(ann)

    result = ann.predict(np.array(inputs[1:3], np.float32))
    print(result)
    print(display_result(result, alphabet))

    result = ann.predict(np.array(inputs[4:6], np.float32))
    print(result)
    print(display_result(result, alphabet))

    result = ann.predict(np.array(inputs[0:60], np.float32))
    print(result)
    print(display_result(result, alphabet))

    return ann


def compute_skew(file_name):
    # load in grayscale:
    src = cv2.imread(file_name, 0)
    height, width = src.shape[0:2]

    # invert the colors of our image:
    cv2.bitwise_not(src, src)

    # Hough transform:
    minLineLength = width / 2.0
    maxLineGap = 20
    lines = cv2.HoughLinesP(src, 1, np.pi / 180, 100, minLineLength, maxLineGap)

    # calculate the angle between each line and the horizontal line:
    angle = 0.0
    nb_lines = len(lines)

    for line in lines:
        angle += math.atan2(line[0][3] * 1.0 - line[0][1] * 1.0, line[0][2] * 1.0 - line[0][0] * 1.0);

    angle /= nb_lines * 1.0

    return angle * 180.0 / np.pi


def deskew(file_name, angle):
    # load in grayscale:
    img = cv2.imread(file_name, 0)

    # invert the colors of our image:
    cv2.bitwise_not(img, img)

    # compute the minimum bounding box:
    non_zero_pixels = cv2.findNonZero(img)
    center, wh, theta = cv2.minAreaRect(non_zero_pixels)

    root_mat = cv2.getRotationMatrix2D(center, angle, 1)
    rows, cols = img.shape
    rotated = cv2.warpAffine(img, root_mat, (cols, rows), flags=cv2.INTER_CUBIC)

    # Border removing:
    sizex = np.int0(wh[0])
    sizey = np.int0(wh[1])

    if theta > -45:
        temp = sizex
        sizex = sizey
        sizey = temp
    return cv2.getRectSubPix(rotated, (sizey, sizex), center)


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
    global cancel
    extracted_text = ""
    # TODO - Izvuci tekst sa ulazne fotografije i vratiti ga kao string

    alphabet = ['A', 'B', 'C', 'Č', 'Ć', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
                'R', 'S', 'Š', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Ž', 'a', 'b', 'c', 'č', 'ć', 'd', 'e',
                'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
                'r', 's', 'š', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ž']
    # Učitavanje slike i određivanje regiona od interesa
    # image_color = load_image(image_path)
    # img = image_bin(image_gray(image_color))
    # selected_regions, letters, distances = select_roi(image_color.copy(), img)
    # display_image(selected_regions)
    # print('Broj prepoznatih regiona:', len(letters))



    inputs = []
    image_color = load_image(image_path)
    display_image(image_color)
    img = invert(image_bin(image_gray(image_color)))
    # display_image(img)
    img_bin = erode(dilate(img))
    # display_image(img_bin)
    # if cancel:
    #     return ""

    # todo 2
    thresh = cv2.threshold(img_bin, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # coords = np.column_stack(np.where(thresh > 0))
    # angle = cv2.minAreaRect(coords)[-1]
    # if angle < -45:
    #     angle = -(90 + angle)
    # else:
    #     angle = -angle
    #
    # (h, w) = image_color.shape[:2]
    # center = (w // 2, h // 2)
    # M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # rotated = cv2.warpAffine(image_color, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    #
    # cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # display_image(rotated)

    # todo 2
    # coords = np.column_stack(np.where(thresh > 0))
    # hgt_rot_angle = cv2.minAreaRect(coords)[-1]
    # com_rot_angle = hgt_rot_angle + 90 if hgt_rot_angle < -45 else hgt_rot_angle
    #
    # (h, w) = image_color.shape[0:2]
    # center = (w // 2, h // 2)
    # M = cv2.getRotationMatrix2D(center, com_rot_angle, 1.0)
    # corrected_image = cv2.warpAffine(image_color, M, (w, h), flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
    # display_image(corrected_image)

    # todo 3
    # file_path =  image_path
    # angel = compute_skew(file_path)
    # dst = deskew(file_path, angel)
    # display_image(dst)

    selected_regions, letters, distances = select_roi(image_color.copy(), img)
    # display_image(numbers[0])
    # if image_path == '.\\dataset\\validation\\train45.png':
    #     cancel = True

    if cancel:
        return ""
    for result in prepare_for_ann(letters):
        inputs.append(result)

    if len(inputs) == 0 or len(inputs) == 1:
        return ""

    if len(letters) == 0 or len(letters) == 1 or len(distances) == 0 or len(distances) == 1:
        return ""

    # Podešavanje centara grupa K-means algoritmom
    distances = np.array(distances).reshape(len(distances), 1)
    # Neophodno je da u K-means algoritam bude prosleđena matrica u kojoj vrste određuju elemente
    k_means = KMeans(n_clusters=2, max_iter=2000, tol=0.00001, n_init=10)
    k_means.fit(distances)

    # inputs = prepare_for_ann(letters)
    results = trained_model.predict(np.array(inputs, np.float32))

    extracted_text = display_result_with_distances(results, alphabet, k_means)
    print(extracted_text)

    return extracted_text
