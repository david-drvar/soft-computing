# import libraries here
import datetime
import cv2
from PIL import Image
import sys

import pyocr
import pyocr.builders
import numpy as np

import matplotlib
import matplotlib.pyplot as plt


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

    plt.imshow(image)
    plt.show()

    text = tool.image_to_string(
        Image.fromarray(image),
        lang=lang,
        builder=pyocr.builders.TextBuilder(tesseract_layout=3)  # izbor segmentacije (PSM)
    )


    # todo izlaz po redovima
    line_and_word_boxes = tool.image_to_string(
        Image.fromarray(image), lang=lang,
        builder=pyocr.builders.LineBoxBuilder(tesseract_layout=3)
    )
    for i, line in enumerate(line_and_word_boxes):
        print('line %d' % i)
        print(line.content, line.position)
        print('boxes')
        for box in line.word_boxes:
            print(box.content, box.position, box.confidence)
        print()

    # todo izlaz lista reči sa tekstom, koordinatama i faktorom sigurnosti
    word_boxes = tool.image_to_string(
        Image.fromarray(image),
        lang=lang,
        builder=pyocr.builders.WordBoxBuilder(tesseract_layout=3)
    )
    for i, box in enumerate(word_boxes):
        print("word %d" % i)
        print(box.content, box.position, box.confidence)
        print()

    # todo detekcija broja
    digits = tool.image_to_string(
        Image.fromarray(image),
        lang=lang,
        builder=pyocr.builders.DigitBuilder(tesseract_layout=3)  # ocekivani text je single line, probati sa 3,4,5..
    )

    # todo rotate image

    canimg = cv2.Canny(gray, 50, 200)
    lines = cv2.HoughLines(canimg, 1, np.pi / 180.0, 250, np.array([]))
    rho, theta = lines[0][0]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 180*theta/3.1415926-90, 1.0)
    newImage = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    plt.imshow(newImage)
    plt.show()
    return person
