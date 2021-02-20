from PIL import Image
import pathlib
import cv2 
import pytesseract
import urllib
import numpy as np
import re
import sys



def rotateImage(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)

    rot_data = pytesseract.image_to_osd(image)
    print("[OSD] "+rot_data)
    rot = re.search('(?<=Rotate: )\d+', rot_data)

    angle = float(rot)
    if angle > 0:
        angle = 360 - angle
    print("[ANGLE] "+str(angle))

    # rotate the image to deskew it
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated	



def RemoveStryLines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(image.shape, dtype=np.uint8)

    cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    cv2.fillPoly(mask, cnts, [255,255,255])
    mask = 255 - mask
    result = cv2.bitwise_or(image, mask)

    return result


def ostsuthresholding(img):
    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

    return th3
    
def sharpenImage(img):
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    im = cv2.filter2D(img, -1, kernel)
    return im

def preprocess_image(path):
    Image = cv2.imread(path)
    img = RemoveStryLines(Image)
    img_final = sharpenImage(img)
    cv2.imwrite('output_images/processed_image.png', img_final) 
