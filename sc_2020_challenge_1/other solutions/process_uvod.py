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
    plt.show()

    # smooth filter
    img_gray = cv2.medianBlur(img_gray, 5)
    plt.imshow(img_gray, 'gray')
    plt.show()

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
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    temp = cv2.erode(img_gray, kernel, iterations=1000)
    # img_open = cv2.dilate(image_edges, kernel, iterations=3)
    plt.imshow(temp, 'gray')
    plt.show()

    # image edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    image_edges = cv2.dilate(img_gray, kernel, iterations=1) - cv2.erode(img_gray, kernel, iterations=1)
    plt.imshow(image_edges, 'gray')
    plt.show()



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
    plt.show()

    # delete sum
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    img_ero = cv2.erode(image_sharpened_bin, kernel, iterations=2)
    img_open = cv2.dilate(image_sharpened_bin, kernel, iterations=3)
    plt.imshow(img_open, 'gray')
    plt.show()


    # circle detection
    # display = cv2.imread(image_path)
    #
    circles = cv2.HoughCircles(img_open, cv2.HOUGH_GRADIENT, 1.42, 200, param1=50, param2=28, minRadius=5, maxRadius=20)
    # Cell_count, x_count, y_count = [], [], []

    # if circles is not None:
    #     # convert the (x, y) coordinates and radius of the circles to integers
    #     circles = np.round(circles[0, :]).astype("int")
    #
    #     # loop over the (x, y) coordinates and radius of the circles
    #     for (x, y, r) in circles:
    #         cv2.circle(display, (x, y), r, (0, 255, 0), 2)
    #         cv2.rectangle(display, (x - 2, y - 2),
    #                       (x + 2, y + 2), (0, 128, 255), -1)
    #         Cell_count.append(r)
    #         x_count.append(x)
    #         y_count.append(y)
    #         # show the output image
    #     cv2.imshow("gray", display)
    #     cv2.waitKey(0)

    # circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
    # circles = np.uint16(np.around(circles))


# TODO CV
#
#     # read original image
#     image = cv2.imread(image_path)
#     plt.imshow(image)
#     plt.show()
#
#     # convet to gray scale image
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     plt.imshow(gray, 'gray')
#     plt.show()
#
#     # apply median filter for smoothning
#     blurM = cv2.medianBlur(gray, 5)
#     plt.imshow(blurM, 'gray')
#     plt.show()
#
#     # apply gaussian filter for smoothning
#     blurG = cv2.GaussianBlur(gray, (9, 9), 0)
#     plt.imshow(blurG, 'gray')
#     plt.show()
#
#     # histogram equalization
#     histoNorm = cv2.equalizeHist(gray)
#     plt.imshow(histoNorm, 'gray')
#     plt.show()
#
#     # create a CLAHE object for
#     # Contrast Limited Adaptive Histogram Equalization (CLAHE)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     claheNorm = clahe.apply(gray)
#     plt.imshow(claheNorm, 'gray')
#     plt.show()
#
#     # contrast stretching
#     # Function to map each intensity level to output intensity level.
#     def pixelVal(pix, r1, s1, r2, s2):
#         if (0 <= pix and pix <= r1):
#             return (s1 / r1) * pix
#         elif (r1 < pix and pix <= r2):
#             return ((s2 - s1) / (r2 - r1)) * (pix - r1) + s1
#         else:
#             return ((255 - s2) / (255 - r2)) * (pix - r2) + s2
#
#             # Define parameters.
#
#     r1 = 70
#     s1 = 0
#     r2 = 200
#     s2 = 255
#
#     # Vectorize the function to apply it to each value in the Numpy array.
#     pixelVal_vec = np.vectorize(pixelVal)
#
#     # Apply contrast stretching.
#     contrast_stretched = pixelVal_vec(gray, r1, s1, r2, s2)
#     contrast_stretched_blurM = pixelVal_vec(blurM, r1, s1, r2, s2)
#
#     plt.imshow(contrast_stretched, 'gray')
#     plt.show()
#
#     plt.imshow(contrast_stretched_blurM, 'gray')
#     plt.show()
#
#     # edge detection using canny edge detector
#     edge = cv2.Canny(gray, 100, 200)
#     plt.imshow(edge, 'gray')
#     plt.show()
#
#     edgeG = cv2.Canny(blurG, 100, 200)
#     plt.imshow(edgeG, 'gray')
#     plt.show()
#
#     edgeM = cv2.Canny(blurM, 100, 200)
#     plt.imshow(edgeM, 'gray')
#     plt.show()





























    # TODO CIRCLES CALCULATION

    # img = cv2.imread(image_path, 0)
    #
    # # morphological operations
    # kernel = np.ones((5, 5), np.uint8)
    # dilation = cv2.dilate(img, kernel, iterations=1)
    # plt.imshow(dilation, 'gray')
    # plt.show()
    # closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # plt.imshow(closing, 'gray')
    # plt.show()
    #
    #
    # # Adaptive thresholding on mean and gaussian filter
    # th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    # plt.imshow(th2, 'gray')
    # plt.show()
    #
    # th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # plt.imshow(th3, 'gray')
    # plt.show()
    #
    # # Otsu's thresholding
    # ret4, th4 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # plt.imshow(th4, 'gray')
    # plt.show()
    #
    # # Initialize the list
    # Cell_count, x_count, y_count = [], [], []
    #
    # # read original image, to display the circle and center detection
    # display = cv2.imread(image_path)
    #
    # # hough transform with modified circular parameters
    # circles = cv2.HoughCircles(th4, cv2.HOUGH_GRADIENT, 1.2, 20,
    #                            param1=50, param2=28, minRadius=1, maxRadius=30)
    #
    # # circle detection and labeling using hough transformation
    # if circles is not None:
    #     # convert the (x, y) coordinates and radius of the circles to integers
    #     circles = np.round(circles[0, :]).astype("int")
    #
    #     # loop over the (x, y) coordinates and radius of the circles
    #     for (x, y, r) in circles:
    #         cv2.circle(display, (x, y), r, (0, 255, 0), 2)
    #         cv2.rectangle(display, (x - 2, y - 2),
    #                       (x + 2, y + 2), (0, 128, 255), -1)
    #         Cell_count.append(r)
    #         x_count.append(x)
    #         y_count.append(y)
    #         # show the output image
    #     cv2.imshow("gray", display)
    #     cv2.waitKey(0)
    #
    #     # display the count of white blood cells
    # print(len(Cell_count))
    # # Total number of radius
    # print(Cell_count)
    # # X co-ordinate of circle
    # print(x_count)
    # # Y co-ordinate of circle
    # print(y_count)






    return red_blood_cell_count, white_blood_cell_count, True
