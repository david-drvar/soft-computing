# import libraries here
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt


def count_blood_cells(image_path):
    """
    Procedura prima putanju do fotografije i vraca broj crvenih krvnih zrnaca, belih krvnih zrnaca i
    informaciju da li pacijent ima leukemiju ili ne, na osnovu odnosa broja krvnih zrnaca

    Ova procedura se poziva automatski iz main procedure i taj deo kod nije potrebno menjati niti implementirati.

    :param image_path: <String> Putanja do ulazne fotografije.
    :return: <int>  Broj prebrojanih crvenih krvnih zrnaca,
             <int> broj prebrojanih belih krvnih zrnaca,
             <bool> da li pacijent ima leukemniju (True ili False)
    """
    red_blood_cell_count = 0
    white_blood_cell_count = 0
    has_leukemia = None

    # TODO - Prebrojati crvena i bela krvna zrnca i vratiti njihov broj kao povratnu vrednost ove procedure

    # TODO - Odrediti da li na osnovu broja krvnih zrnaca pacijent ima leukemiju i vratiti True/False kao povratnu vrednost ove procedure


    # WHITE COLOR - OBJECT OF INTEREST
    # BLACK - BACKGROUND

    # erode - slim white
    # dilate - expand white

    # my code
    red_blood_cells = []
    white_blood_cells = []
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_gray = img
    plt.imshow(img, 'gray')
    # plt.show()

    # smooth filter
    img_gray = cv2.medianBlur(img_gray, 5)
    plt.imshow(img_gray, 'gray')
    # plt.show()

###############################
    # image_bin = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5) #cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 10)
    #
    # # plt.imshow(image_bin, 'gray')
    # # plt.show()
    #
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # img_delated = cv2.dilate(image_bin, kernel, iterations=1)
    # img_closed = cv2.erode(img_delated, kernel, iterations=3)
    #
    # # plt.imshow(img_closed, 'gray')
    # # plt.show()
###############################################
    # sharpening
    # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    # img_gray = cv2.filter2D(img_gray, -1, kernel)
    # plt.imshow(img_gray, 'gray')
    # plt.show()

    # try deleting red blood cells
    img = cv2.imread(image_path)


    # image edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    image_edges = cv2.dilate(img_gray, kernel, iterations=1) - cv2.erode(img_gray, kernel, iterations=1)
    plt.imshow(image_edges, 'gray')
    # plt.show()



    # # sharpening
    # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    # img_sharpened = cv2.filter2D(image_edges, -1, kernel)
    # plt.imshow(img_sharpened, 'gray')
    # plt.show()

    # fix for other images
    # image_edges = cv2.medianBlur(image_edges, 5)
    # plt.imshow(image_edges, 'gray')
    # plt.show()
    # image_edges = cv2.GaussianBlur(image_edges, (1, 1), 0)
    # plt.imshow(image_edges, 'gray')
    # plt.show()

    # binary
    ret, image_sharpened_bin = cv2.threshold(image_edges, 0, 255, cv2.THRESH_OTSU)
    plt.imshow(image_sharpened_bin, 'gray')
    # plt.show()

    # delete noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    img_ero = cv2.erode(image_sharpened_bin, kernel, iterations=2)
    img_open = cv2.dilate(image_sharpened_bin, kernel, iterations=3)
    plt.imshow(img_open, 'gray')
    # plt.show()


    # circle detection
    circles = cv2.HoughCircles(img_open, cv2.HOUGH_GRADIENT, 1.42, 200, param1=50, param2=28, minRadius=5, maxRadius=20)

    return red_blood_cell_count, circles.size, False
