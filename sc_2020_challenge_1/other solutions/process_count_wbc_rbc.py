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




    # contrast tweak
    # img = cv2.imread(image_path)
    # plt.imshow(img)
    # plt.show()
    #
    # #blur
    # img = cv2.medianBlur(img, 5)
    # plt.imshow(img)
    # plt.show()
    #
    # #binary
    # # ret, image_sharpened_bin = cv2.threshold(image_edges, 0, 255, cv2.THRESH_OTSU)
    # # plt.imshow(image_sharpened_bin, 'gray')
    #
    # # -----Converting image to LAB Color model-----------------------------------
    # lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # cv2.imshow("lab", lab)
    #
    # # -----Splitting the LAB image to different channels-------------------------
    # l, a, b = cv2.split(lab)
    # cv2.imshow('l_channel', l)
    # cv2.imshow('a_channel', a)
    # cv2.imshow('b_channel', b)
    #
    # # -----Applying CLAHE to L-channel-------------------------------------------
    # clahe = cv2.createCLAHE(clipLimit=100.0, tileGridSize=(1, 1))
    # cl = clahe.apply(l)
    # cv2.imshow('CLAHE output', cl)
    #
    # # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    # limg = cv2.merge((cl, a, b))
    # cv2.imshow('limg', limg)
    #
    # # -----Converting image from LAB Color model to RGB model--------------------
    # final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    # plt.imshow(final)
    # plt.show()
    #
    # # binary
    # final = cv2.cvtColor(final, cv2.COLOR_RGB2GRAY)
    # image_bin = cv2.adaptiveThreshold(final, 255, cv2.OPTFLOW_FARNEBACK_GAUSSIAN, cv2.THRESH_BINARY, 15, 5)
    # plt.imshow(image_bin)
    # plt.show()


#todo CANNY
    img = cv2.imread(image_path)
    plt.imshow(img)
    #plt.show()
    original = img



    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    plt.imshow(img, 'gray')
    #plt.show()

    # hist_full = cv2.calcHist([img], [0], None, [255], [0, 255])
    # plt.plot(hist_full)
    # plt.show()


    img = cv2.GaussianBlur(img,(9,9),0)

    #sharpening
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    img = cv2.filter2D(img, -1, kernel)



    edges = cv2.Canny(img,50,50)
    plt.imshow(edges, 'gray')
    #plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img_ero = cv2.dilate(edges, kernel, iterations=1)
    img_open = cv2.erode(img_ero, kernel, iterations=1)
    plt.imshow(img_open, 'gray')
    #plt.show()

    # circles = cv2.HoughCircles(edges,cv2.HOUGH_GRADIENT,dp=1,minDist=50,param1=18,param2=8,minRadius=0,maxRadius=0)
    #
    #
    #
    # for i in circles[0, :]:
    #     #	draw	the	outer	circle
    #     cv2.circle(original, (i[0], i[1]), i[2], (0, 255, 0), 6)
    #     #	draw	the	center	of	the	circle
    #     cv2.circle(original, (i[0], i[1]), 2, (0, 0, 255), 3)
    #
    #
    #
    # plt.imshow(original)
    # plt.show()

    inverted_binary = cv2.bitwise_not(edges)
    plt.imshow(inverted_binary, 'gray')
    #plt.show()

    (_, contours, _) = cv2.findContours(inverted_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    a = len(contours)

    img, contours, hierarchy = cv2.findContours(inverted_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    b = len(contours)

    circle_countours = []
    for contour in contours:  # za svaku konturu
        center, size, angle = cv2.minAreaRect(contour)  # pronadji pravougaonik minimalne povrsine koji ce obuhvatiti celu konturu
        width, height = size
        answer = cv2.isContourConvex(contour)
        # if width > 30 and width < 80 and height > 30 and height < 80:  # uslov da kontura pripada bar-kodu
        #     circle_countours.append(contour)  # ova kontura pripada bar-kodu
        if cv2.isContourConvex(contour):
            circle_countours.append(contour)

    img = cv2.imread(image_path)
    cv2.drawContours(img, circle_countours, -1, (255, 0, 0), 1)
    plt.imshow(img)
    #plt.show()



    # todo kmeans

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.medianBlur(img, 5)

    plt.imshow(img, 'gray')
    plt.show()

    Z = img.reshape((-1, 3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    plt.imshow(res2, 'gray')
    plt.show()


    # ret, res2 = cv2.threshold(res2, 0, 255, cv2.THRESH_OTSU)
    # # res2 = cv2.adaptiveThreshold(res2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)
    # plt.imshow(res2, 'gray')
    # plt.show()


    #globalni threshold
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



    # todo find contours on black mask - red cells
    img, contours_red, hierarchy = cv2.findContours(global_thres, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    red_cells = len(contours_red)

    circle_countours = []
    for contour in contours_red:
        center, size, angle = cv2.minAreaRect(contour)
        width, height = size
        if width > 30 and width < 300 and height > 30 and height < 300:
            circle_countours.append(contour)


    img = cv2.imread(image_path)
    cv2.drawContours(img, circle_countours, -1, (255, 0, 0), 1)
    plt.imshow(img)
    plt.show()


    # todo  count white cells
    # global_thres = res2 > index - 4
    # plt.imshow(global_thres, 'gray')
    # plt.show()
    mask = cv2.inRange(res2, index - 5, index + 5)
    mask = cv2.copyMakeBorder(mask, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255,255,255))
    plt.imshow(mask, 'gray')
    plt.show()
    img, countours_white, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    circle_countours_white = []
    # for contour in countours_white:
    #     center, size, angle = cv2.minAreaRect(contour)
    #     width, height = size
    #     #if width > 30 and width < 300 and height > 30 and height < 300:
    #     circle_countours_white.append(contour)
    for i in range(1, len(countours_white)): # range 1 because root contour is excluded
        # if hierarchy[0,i,3] == -1:
        circle_countours_white.append(countours_white[i])

    # circles_final = []
    # for i in range(len(circle_countours_white)):
    #     if hierarchy[0,i,3] != -1:
    #         circles_final.append(countours_white[i])

    #circle_countours_white.__delitem__(circle_countours_white[0])


    img = cv2.imread(image_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(mask, circle_countours_white, -1, (255, 0, 0), 1)
    plt.imshow(mask)
    plt.show()




    hist_full = cv2.calcHist([res2], [0], None, [255], [0, 255])
    #plt.imshow(hist_full)
    #plt.show()


    # try masking
    mask = np.zeros(res2.shape[:2], np.uint8)
    # a = len(res2)
    # row = res2[0]
    # columns = len(row)
    max_hist = max(hist_full)
    index = 0
    for array in hist_full:
        if max(array) == max_hist:
            break
        index = index + 1

    #plt.imshow(res2, 'gray')
    #plt.show()

    row_index = 0
    column_index = 0
    for row in res2:
        column_index = 0
        for column in row:
            if column == index-1 or column == index or column == index + 1:
                mask[row_index][column_index] = 1
            column_index = column_index + 1
        row_index = row_index + 1

    # res2[mask == 255] = 0
    plt.imshow(mask, 'gray')
    #plt.show()

    fully = cv2.bitwise_and(res2, res2, mask=mask)
    plt.imshow(fully)
    #plt.show()

    # res2 = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)


    mask = cv2.inRange(res2, index-5, index+5)
    plt.imshow(mask, 'gray')
    #plt.show()


    fully = cv2.bitwise_and(res2, mask, mask=mask)
    plt.imshow(fully, 'gray')
    #plt.show()
    # fully = cv2.bitwise_and(res2, res2, mask=mask)
    # plt.plot(fully)
    # plt.show()

    edges = cv2.Canny(res2, 50, 50)
    plt.imshow(edges, 'gray')
    #plt.show()

    inverted_binary = cv2.bitwise_not(edges)
    # plt.imshow(inverted_binary, 'gray')
    # plt.show()

    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    inverted_binary = cv2.filter2D(inverted_binary, -1, kernel)
    plt.imshow(inverted_binary, 'gray')
    #plt.show()

    #drawing contours
    # img, contours, hierarchy = cv2.findContours(inverted_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # b = len(contours)
    #
    # circle_countours = []
    # for contour in contours:
    #     if cv2.isContourConvex(contour):
    #         circle_countours.append(contour)
    #
    # img = cv2.imread(image_path)
    # cv2.drawContours(img, circle_countours, -1, (255, 0, 0), 1)
    # plt.imshow(img)
    #plt.show()

    return len(circle_countours), len(circle_countours_white) - len(circle_countours), False
