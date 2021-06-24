"""UTILS
Misc helpers/utils functions
"""

# # Native # #
import re
import math
from time import time
from uuid import uuid4
from typing import Union
from operator import itemgetter
from PIL import Image as im
import io
import json
import numpy as np

# # Installed # #
import pandas as pd
import requests
from PyPDF2 import PdfFileReader
import cv2
import pytesseract
from scipy.ndimage import interpolation as inter
from nltk import tokenize
import nltk
from fuzzywuzzy import process
from nltk.tokenize import word_tokenize 
import spacy
from string import punctuation
import tabula
nlp = spacy.load("en_core_web_sm")

stop_words = nlp.Defaults.stop_words
#pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

__all__ = ("get_time", "get_uuid", "ocr_extraction", "aadhar_card_info", 'prescription_info', 'report_info')

def get_time(seconds_precision=True) -> Union[int, float]:
    """Returns the current time as Unix/Epoch timestamp, seconds precision by default"""
    return time() if not seconds_precision else int(time())


def get_uuid() -> str:
    """Returns an unique UUID (UUID4)"""
    return str(uuid4())

def keyword_extraction(sentence):
    sentence = re.sub(r'[^\w\s]', '', sentence)
    result = []

    # custom list of part of speech tags we are interested in
    # we are interested in proper nouns, nouns, and adjectives
    # edit this list of POS tags according to your needs. 
    pos_tag = ['PROPN','NOUN','ADJ']

    # create a spacy doc object by calling the nlp object on the input sequence
    doc = nlp(sentence)

    for chunk in doc.noun_chunks:
        final_chunk = ""
        for token in chunk:
            if (token.pos_ in pos_tag):
                final_chunk =  final_chunk + token.text + " "
        if final_chunk:
            result.append(final_chunk.strip())


    for token in doc:
        if (token.text in stop_words or token.text in punctuation):
            continue
        if (token.pos_ in pos_tag):
            result.append(token.text)
    
    return ' '.join(list(set(result)))

def top_10_symptom_match(symptom, symptoms_list):
    symptom = symptom.lower()
    top_10_symptom_match = process.extract(symptom, symptoms_list, limit=10)
    return top_10_symptom_match

def diseaseToSymptom(symptom):
    df = pd.read_json('./Data/disease_data.json')
    symptoms_list = []
    for i in df.index:
        symptoms_list.append(keyword_extraction(' '.join(df['Symptoms'][i])))
    symptoms_list = pd.Series(symptoms_list)
    matches = top_10_symptom_match(keyword_extraction(symptom), symptoms_list)
    diseaseList = []
    for symptom in matches:
        diseaseList.append(
            {
                'name': df["Name"][symptom[2]],
                'introduction': df["Introduction"][symptom[2]],
                'symptom': df["Symptoms"][symptom[2]],
                'cause': df["Causes"][symptom[2]],
                'diagnosis': df["Diagnosis"][symptom[2]],
                'cure': df["Management"][symptom[2]],
                'score': symptom[1]
            }
        ) 
    return diseaseList

def correct_skew(image, delta=1, limit=5):
  def determine_score(arr, angle):
    data = inter.rotate(arr, angle, reshape=False, order=0)
    histogram = np.sum(data, axis=1)
    score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
    return histogram, score

  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 

  scores = []
  angles = np.arange(-limit, limit + delta, delta)

  for angle in angles:
    histogram, score = determine_score(thresh, angle)
    scores.append(score)

  best_angle = angles[scores.index(max(scores))]

  (h, w) = image.shape[:2]
  center = (w // 2, h // 2)
  M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
  rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
              borderMode=cv2.BORDER_REPLICATE)

  return best_angle, rotated


def ocr_extraction(image):
    img = im.open(image.file)
    img.thumbnail(size=(2048,2048))
    img_raw = img
    
    raw_image = np.array(img)
    raw_image = raw_image[:, :, ::-1].copy()

    angle, skew_image = correct_skew(raw_image)
    skew_image_raw = skew_image

    orig_image = np.array(img_raw)
    orig_image = orig_image[:,:,::-1].copy()
    orig_image_raw = orig_image

    extracted_info_from_skew = pytesseract.image_to_string(cv2.medianBlur(skew_image_raw,3))

    extracted_info_from_orig = pytesseract.image_to_string(cv2.medianBlur(orig_image_raw,1))
    
    extracted_info = extracted_info_from_skew + " string_separation_between_two_extraction " + extracted_info_from_orig
    #print(extracted_info)

    return extracted_info

def aadhar_card_info(aadhar_card):
    info = ocr_extraction(aadhar_card)
    n = len(info)
    aadhar_len = 12 + 2
    aadhaar = 0
    start = -1
    for i in range(n - aadhar_len):
        current = 0
        for j in range(aadhar_len):
            char = info[i + j]
            if j == 4 and char >= ' ':
                current += 1
            elif j == 9 and char >= ' ':
                current += 1
            elif j != 4 and j != 9 and char.isdigit():
                current += 1
        if i > 0 and (info[i - 1] == ' ' or info[i - 1] == '\n'):
            current += 1
        elif not i:
            current += 1
        if i + aadhar_len < n and info[i + aadhar_len] == ' ' or info[i + aadhar_len] == '\n':
            current += 1
        elif i == n-aadhar_len:
            current += 1
        if current >= 14:
            aadhaar = 1
            start = i
            break
    if aadhaar:
        aadhar_number = ''.join(info[start : start+aadhar_len].split())
        return aadhar_number
    return "Not found!"
def get_useful_info(info):
    info = keyword_extraction(info)
    diseases = pd.read_json('./Data/disease_data.json')
    disease_name = diseases['Name']
    
    medicine = pd.read_csv('./Data/medicine_database_detailed.csv')
    medicine_name = medicine['Medicine Name']
    
    disease_found = process.extract(info, disease_name, limit=5)
    medicine_found = process.extract(info, medicine_name, limit=5)

    return {
        'diseases' : disease_found,
        'medicine' : medicine_found,
        'extracted_info': info
    }

def prescription_info(prescription):
    extracted_info = ocr_extraction(prescription).split("string_separation_between_two_extraction")
    extracted_info = max(extracted_info, key=len)
    return get_useful_info(extracted_info)
    

def report_info(reportUrl):
    
    dfs = tabula.read_pdf(reportUrl, stream=True, pages = 'all')
    print(len(dfs))
    report = pd.concat(dfs, ignore_index=True)
    print(report)
    r = requests.get(reportUrl)
    f = io.BytesIO(r.content)

    reader = PdfFileReader(f)

    text = ""
    for i in range(reader.numPages):
        text += reader.getPage(i).extractText() + " "
    
    info = ' '.join(list(set(text.split())))
    data =  diseaseToSymptom(info)
    diseases = [{"name" :d['name'], "score" : d["score"]} for d in data]
        
    return {
        "data": json.loads(report.to_json(orient='records')),
        "disease": diseases
    }