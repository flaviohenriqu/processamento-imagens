"""

@author: Flavio Henrique
"""
import cv2
import numpy as np
from PIL import Image
from PIL import ImageOps
import os
import tempfile

total = 0
cont = 0
PATH = os.getcwd()


def segmentation(path, name):
    global total, cont

    path_full = PATH + path + name

    if not os.path.exists(path_full):
        print('file not found.')
        return

    with open(path_full) as f:
        for item in f.read().split('\n'):
            image = ''.join(PATH, path, item)
            if os.path.exists(image):
                tmp = process(image)
                cv2.imwrite(tempfile.TemporaryFile(), tmp)

        
def process(img, debug=False):

    img_debug = []
    
    img_a = cv2.imread(img)
    img_debug.append(img_a.copy())

    # step 1
    img_b = np.asarray(img_a.dot([1.900, -0.301, -0.599]), dtype="uint8")
    img_debug.append(img_b.copy())

    # step 2
    img_l = np.asarray(ImageOps.autocontrast(Image.fromarray(np.uint8(img_b))))
    img_debug.append(img_l.copy())

    # step 3
    img_h = cv2.equalizeHist(img_b)
    img_debug.append(img_h.copy())

    # step 4
    img_r1 = cv2.add(img_l, img_h)
    img_debug.append(img_r1.copy())
    # step 5
    img_r2 = cv2.subtract(img_l, img_h)
    img_debug.append(img_r2.copy())
    # step 6
    img_r3 = cv2.add(img_r1, img_r2)
    img_debug.append(img_r3.copy())

    # step 7
    for i in range(3):
        img_r3 = cv2.blur(img_r3, (3, 3))

    img_debug.append(img_r3.copy())
    # steps 8 and 9
    thresh = method_otsu(img_r3)
    
    img_r3 = Image.fromarray(np.uint8(img_r3))
    
    result = img_r3.point(lambda func: func >= thresh and 255)
    img_debug.append(np.asarray(result.copy(), dtype="uint8"))

    # step 10
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))

    opening = cv2.morphologyEx(np.asarray(result.copy(), dtype="uint8"), cv2.MORPH_OPEN, kernel)
    img_debug.append(opening.copy())

    # step 11
    closing = cv2.morphologyEx(opening.copy(), cv2.MORPH_CLOSE, kernel)
    img_debug.append(closing.copy())
    
    im = closing
    # step 12
    calc_area(im)
    img_debug.append(im.copy())
    
    im = cv2.bitwise_and(img_a, cv2.cvtColor(im, cv2.COLOR_GRAY2BGR))
    if debug:
        show(img_debug, debug)
            
    return im


def area_media(img):
    global total
    
    max_area = -np.inf
    contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > max_area:
            max_area = cv2.contourArea(cnt)
            
    total += max_area
    

def calc_area(img):
    cnt, h = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cnt = [c for c in cnt if cv2.contourArea(c) < 4000.0]

    cv2.drawContours(img, cnt, 0, 255, -1)


def method_otsu(img):

    blur = cv2.GaussianBlur(img, (5, 5), 0)

    # find normalized_histogram, and your cumulative distribution function
    hist = cv2.calcHist([blur], [0], None, [256], [0, 256])
    hist_norm = hist.ravel()/hist.max()
    Q = hist_norm.cumsum()

    bins = np.arange(256)

    fn_min = np.inf
    thresh = -1

    for i in xrange(1, 256):
        # probability
        p1, p2 = np.hsplit(hist_norm, [i])
        # cumulative distribution function
        q1, q2 = Q[i], Q[255]-Q[i]
        # weights
        b1, b2 = np.hsplit(bins, [i])

        if q1 == 0:
            continue

        if q2 == 0:
            break
                
        # finding averages and variances
        m1, m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
        v1, v2 = np.sum(((b1-m1)**2)*p1)/q1, np.sum(((b2-m2)**2)*p2)/q2

        # calculates the function minimization
        fn = v1*q1 + v2*q2
        if fn < fn_min:
            fn_min = fn
            thresh = i

    return thresh


def show(img, debug=False):
    if debug:
        cont = 0
        for v in img:
            cont += 1
            name = "image" + str(cont)
            cv2.imshow(name, v)
        cv2.waitKey(0)        
    else:
        cv2.imshow("default", img)
        cv2.waitKey(0)
