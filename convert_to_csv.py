#Helper function to convert extracted table image into a csv file
#Doesnt work that great (just a backup method)


import random
import os
from os import listdir
from xml.etree import ElementTree
import cv2
import glob
from PIL import Image
from random import randrange
import numpy as np
import pytesseract


def convert(image):
    i=cv2.read(image_path)
    gray_image = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
    threshold_img = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    kernel = np.ones((1, 1), np.uint8)
    threshold_img = cv2.dilate(threshold_img, kernel, iterations=1)
    threshold_img = cv2.erode(threshold_img, kernel, iterations=1)

    #configuring parameters for tesseract
    from pytesseract import Output
    custom_config = r'--oem 3 --psm 6'

    # now feeding image to tesseract

    details = pytesseract.image_to_data(threshold_img, output_type=Output.DICT,config=custom_config)

    print(details.keys())
    from pytesseract import Output
    custom_config = r'--oem 3 --psm 6'
    total_boxes = len(details['text'])

    for sequence_number in range(total_boxes):

        if int(details['conf'][sequence_number]) >5:
            (x, y, w, h) = (details['left'][sequence_number], details['top'][sequence_number], details['width'][sequence_number],  details['height'][sequence_number])
            threshold_img = cv2.rectangle(threshold_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # display image


    # now feeding image to tesseract

    details = pytesseract.image_to_data(threshold_img, output_type=Output.DICT,config=custom_config)
    parse_text = []

    word_list = []

    last_word = ''

    for word in details['text']:

        if word!='':

            word_list.append(word)

            last_word = word

        if (last_word!='' and word == '') or (word==details['text'][-1]):

            parse_text.append(word_list)

            word_list = []
    import csv
    # saving the extracted text output to a txt file

    with open('result.txt','w', newline="") as file:

        csv.writer(file, delimiter=" ").writerows(parse_text)

    import pandas as pd
    # reading the txt file into a dataframe to convert to csv file 
    df = pd.read_csv("result.txt",delimiter='\t')
    df.to_csv('result.csv')