import flavio_henrique as fla
import cv2
from numpy import *
from PIL import Image,ImageOps
"""
escala de cinza [2.200, -0.601, -0.599]
cv2.contourArea(c) < 1000.0

BloodImage_00017.jpg
BloodImage_00045.jpg
BloodImage_00048.jpg
BloodImage_00054.jpg
BloodImage_00059.jpg
BloodImage_00062.jpg
BloodImage_00069.jpg
BloodImage_00106.jpg
BloodImage_00164.jpg
BloodImage_00352.jpg


escala de cinza [1.900, -0.301, -0.599]
cv2.contourArea(c) < 1000.0

BloodImage_00034.jpg
BloodImage_00189.jpg
BloodImage_00201.jpg
BloodImage_00239.jpg
BloodImage_00298.jpg
BloodImage_00331.jpg
BloodImage_00392.jpg
BloodImage_00398.jpg


"""
name = "BloodImage_00398.jpg"
path_tmp = "tmp"
img = "img\\BloodImageSetS6NucSeg\\" + name

A = cv2.imread(img)

B = asarray(A.dot([1.900, -0.301, -0.599]),dtype="uint8") # passo 1
#B = cv2.cvtColor(A,cv2.COLOR_BGR2GRAY)
# [2.200, -0.601, -0.599]
# [1.900, -0.301, -0.599]
# [1.905, -0.432, -0.473]
# [1.905, 0.082, -0.987]
# [0.299, 0.587, 0.114]
        
L = asarray(ImageOps.autocontrast(Image.fromarray(uint8(B)))) # passo 2
#L = cv2.multiply(B,asarray([2.0]))

H = cv2.equalizeHist(B) # passo 3

R1 = cv2.add(L,H) # passo 4
R2 = cv2.subtract(L,H) # passo 5
R3 = cv2.add(R1,R2) # passo 6

for i in range(3): #passo 7
    R3 = cv2.blur(R3,(3,3))    

#passos 8 e 9
"""
    thresh, otsu = cv2.threshold(R3,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    rows, cols = np.shape(R3)
    hist = np.histogram(R3,256)[0]
    thresh = otsu2(hist,rows*cols)
"""
thresh = fla.method_otsu(R3)
#thresh, otsu = cv2.threshold(R3,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
R3 = Image.fromarray(uint8(R3))
    
result = R3.point(lambda i: i >= thresh and 255)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(19,19)) #passo 10
"""
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
           [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
           [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)

"""

opening = cv2.morphologyEx(asarray(result,dtype="uint8"),cv2.MORPH_OPEN,kernel)

#opening2 = cv2.morphologyEx(otsu,cv2.MORPH_OPEN,kernel)

closing = cv2.morphologyEx(opening,cv2.MORPH_CLOSE,kernel) #passo 11

#closing2 = cv2.morphologyEx(opening2,cv2.MORPH_CLOSE,kernel) #passo 11

im = closing
#im = fla.calc_area(closing)   #passo 12


cnt, h = cv2.findContours(im.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

cnt = [c for c in cnt if cv2.contourArea(c) < 1000.0]

cv2.drawContours(im, cnt, 0, 255, -1)
"""
im = fla.calc_area(closing2)   #passo 12

for c in cnt:
    print cv2.contourArea(c)

print cnt
"""
im = cv2.bitwise_and(A,cv2.cvtColor(im,cv2.COLOR_GRAY2BGR))

fla.show(B)

#cv2.imwrite(path_tmp+"\\"+name.split(".")[0]+"_tmp."+name.split(".")[1],im)


