# import libraries here
import datetime
import cv2
from PIL import Image
import sys

import pyocr
import pyocr.builders
import numpy as np

import imutils
import dlib

import matplotlib
import matplotlib.pyplot as plt
from imutils import face_utils


class Person:
    """
    Klasa koja opisuje prepoznatu osobu sa slike. Neophodno je prepoznati samo vrednosti koje su opisane u ovoj klasi
    """

    def __init__(self, name: str = None, date_of_birth: datetime.date = None, job: str = None, ssn: str = None,
                 company: str = None):
        self.name = name
        self.date_of_birth = date_of_birth
        self.job = job
        self.ssn = ssn
        self.company = company


def findparallel(lines):
    parallel_lines = []
    for line in lines:
        rho1, theta1 = line[0]
        a = np.cos(theta1)
        b = np.sin(theta1)
        x0 = a * rho1
        y0 = b * rho1
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / 3.14
        for line2 in lines:
            rho2, theta2 = line[0]
            a = np.cos(theta2)
            b = np.sin(theta2)
            x0 = a * rho2
            y0 = b * rho2
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            angle2 = np.arctan2(y2 - y1, x2 - x1) * 180.0 / 3.14
            if angle == angle2:
                parallel_lines.append(line2)

    return parallel_lines


def findparallel_web(lines):
    lines1 = []
    for i in range(len(lines)):
        for j in range(len(lines)):
            if i == j: continue
            a = lines[i][0][1]
            b = lines[j][0][1]
            if abs(lines[i][0][1] - lines[j][0][1]) == 0:
                # You've found a parallel line!
                lines1.append(lines[i])

    return lines1

def is_similar(image1, image2):
    return image1.shape == image2.shape and not(np.bitwise_xor(image1,image2).any())

def extract_info(models_folder: str, image_path: str) -> Person:
    """
    Procedura prima putanju do foldera sa modelima, u slucaju da su oni neophodni, kao i putanju do slike sa koje
    treba ocitati vrednosti. Svi modeli moraju biti uploadovani u odgovarajuci folder.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param models_folder: <str> Putanja do direktorijuma sa modelima
    :param image_path: <str> Putanja do slike za obradu
    :return:
    """
    person = Person('test', datetime.date.today(), 'test', 'test', 'test')

    tools = pyocr.get_available_tools()
    if len(tools) == 0:
        print("No OCR tool found")
        sys.exit(1)

    # odaberemo Tessract - prvi na listi ako je jedini alat
    tool = tools[0]
    print("Koristimo backend: %s" % (tool.get_name()))
    # biramo jezik očekivanog teksta
    lang = 'eng'

    # TODO - Prepoznati sve neophodne vrednosti o osobi sa slike. Vrednosti su: Name, Date of Birth, Job,
    #       Social Security Number, Company Name

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    original = image.copy()
    plt.imshow(image)
    plt.show()

    # todo rotate image
    canimg = cv2.Canny(gray, 50, 200)
    lines = cv2.HoughLines(canimg, 1, np.pi / 180.0, 200, np.array([]))
    rho, theta = lines[0][0]

    plt.imshow(image)
    #plt.show()

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 180 * theta / 3.1415926 - 90, 1.0)
    newImage = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    plt.imshow(newImage)
    plt.show()

    if is_similar(original, newImage):
        rho, theta = findparallel_web(lines)[0][0]
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, 180 * theta / 3.1415926 - 90, 1.0)
        newImage = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        plt.imshow(newImage)
        plt.show()

    freshNewImage = newImage.copy()

    # todo ekstrakcije kartice
    canimg = cv2.Canny(cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY), 50, 200)
    lines = cv2.HoughLines(canimg, 1, np.pi / 180.0, 180, np.array([]))
    min_x = 999
    max_x = 1
    min_y = 999
    max_y = 1
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(newImage, (x1, y1), (x2, y2), (0, 0, 255), 2)

        if 0 < y1 < 1000 and y1 > max_y:
            max_y = y1
        if 0 < y1 < 1000 and y1 < min_y:
            min_y = y1
        if 0 < x1 < 1000 and x1 < min_x:
            min_x = x1
        if 0 < x1 < 1000 and x1 > max_x:
            max_x = x1

    plt.imshow(newImage)
    # plt.show()

    height, width, channels = newImage.shape

    if min_x == 999:
        min_x = 0
    if max_x == 1:
        max_x = width
    if min_y == 999:
        min_y = 0
    if max_y == 1:
        max_y = height
    if min_y == max_y:
        min_y = 0
        max_y = height
    if min_x == max_x:
        min_x = 0
        max_x = width
    crop_img = freshNewImage[min_y:max_y, min_x:max_x]

    plt.imshow(crop_img)
    # plt.show()

    canimg = cv2.Canny(cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY), 50, 200)
    plt.imshow(canimg)
    # plt.show()

    # todo ekstrakcija teksta
    text = tool.image_to_string(
        Image.fromarray(canimg),
        lang=lang,
        builder=pyocr.builders.TextBuilder(tesseract_layout=3)  # izbor segmentacije (PSM)
    )

    return person


    # # todo detekcija broja
    # digits = tool.image_to_string(
    #     Image.fromarray(image),
    #     lang=lang,
    #     builder=pyocr.builders.DigitBuilder(tesseract_layout=3)  # ocekivani text je single line, probati sa 3,4,5..
    # )

    # # todo izlaz lista reči sa tekstom, koordinatama i faktorom sigurnosti
    # word_boxes = tool.image_to_string(
    #     Image.fromarray(image),
    #     lang=lang,
    #     builder=pyocr.builders.WordBoxBuilder(tesseract_layout=3)
    # )
    # for i, box in enumerate(word_boxes):
    #     print("word %d" % i)
    #     print(box.content, box.position, box.confidence)
    #     print()


    # # todo izlaz po redovima
    # line_and_word_boxes = tool.image_to_string(
    #     Image.fromarray(image), lang=lang,
    #     builder=pyocr.builders.LineBoxBuilder(tesseract_layout=3)
    # )
    # for i, line in enumerate(line_and_word_boxes):
    #     print('line %d' % i)
    #     print(line.content, line.position)
    #     print('boxes')
    #     for box in line.word_boxes:
    #         print(box.content, box.position, box.confidence)
    #     print()