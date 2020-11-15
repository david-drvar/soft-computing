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

    # todo kmeans



    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()

    # blur and gray
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.medianBlur(img, 3)
    plt.imshow(img, 'gray')
    plt.show()


    # k means
    Z = img.reshape((-1, 3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 1.0)
    K = 4
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 100, cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    plt.imshow(res2, 'gray')
    plt.show()

    # global threshold
    hist_full = cv2.calcHist([res2], [0], None, [255], [0, 255])

    max_hist = max(hist_full)
    index = 0
    for array in hist_full:
        if max(array) == max_hist:
            break
        index = index + 1



    ret, global_thres = cv2.threshold(res2, 160, 255, cv2.THRESH_BINARY)
    plt.imshow(global_thres, 'gray')
    plt.show()



    # dilate global_thres
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img_delated = cv2.dilate(global_thres, kernel, iterations=1)
    img_eroded = cv2.erode(img_delated, kernel,iterations=1)
    # plt.imshow(img_delated, 'gray')
    # plt.show()

    # todo white cells
    img_eroded = cv2.bitwise_not(img_eroded)
    img, contours_white, hierarchy = cv2.findContours(img_eroded, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


    circle_countours_white = []
    for contour in contours_white:
        center, size, angle = cv2.minAreaRect(contour)
        width, height = size
        if width > 30 and width < 300 and height > 30 and height < 300:
            circle_countours_white.append(contour)

    white_blood_cell_count = len(circle_countours_white)

    img = cv2.imread(image_path)
    cv2.drawContours(img, circle_countours_white, -1, (255, 0, 0), 1)
    plt.imshow(img)
    plt.show()

    # todo red cells
    # global_thres = res2 > index - 4
    # plt.imshow(global_thres, 'gray')
    # plt.show()

    mask = cv2.inRange(res2, index - 100, index-2)
    plt.imshow(mask, 'gray')
    #plt.show()

    # bitwise not mask
    mask = cv2.bitwise_not(mask)
    plt.imshow(mask, 'gray')
    #plt.show()

    # dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    img_delated = cv2.dilate(mask, kernel, iterations=1)
    plt.imshow(img_delated, 'gray')
    #plt.show()

    mask = cv2.copyMakeBorder(mask, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    plt.imshow(mask, 'gray')
    #plt.show()

    img, contours_red, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    circle_countours_red = []
    for i in range(1, len(contours_red)):  # range 1 because root contour is excluded
        # if hierarchy[0,i,3] == -1:
        center, size, angle = cv2.minAreaRect(contours_red[i])
        width, height = size
        if width > 15 and width < 300 and height > 15 and height < 300:
            circle_countours_red.append(contours_red[i])

    red_blood_cell_count = len(circle_countours_red) - white_blood_cell_count

    img = cv2.imread(image_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(mask, circle_countours_red, -1, (255, 0, 0), 1)
    plt.imshow(mask)
    #plt.show()



    has_leukemia = False
    if white_blood_cell_count > 3:
        has_leukemia = True


    # original = cv2.imread(image_path)



    return red_blood_cell_count, white_blood_cell_count, has_leukemia


    # todo ideje
    # kada napravis masku uradi dilaciju/eroziju da smanjis crno i razdvojis neke krugove potencijalno

    # uradi HoughCircles na masku

    # skontaj kako da izbrises konturu u konturi
