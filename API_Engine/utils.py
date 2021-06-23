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
import numpy as np

# # Installed # #
import pandas as pd
import cv2
import pytesseract
from scipy.ndimage import interpolation as inter
from nltk import tokenize
import nltk
from fuzzywuzzy import process
from nltk.tokenize import word_tokenize 

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

__all__ = ("get_time", "get_uuid", "ocr_extraction", "aadhar_card_info")

stop_words = {"a","about","above","after","again","against","ain","all","am","an","and","any","are","aren","aren't","as","at","be","because","been","before","being","below","between","both","but","by","can","couldn","couldn't","d","did","didn","didn't","do","does","doesn","doesn't","doing","don","don't","down","during","each","few","for","from","further","had","hadn","hadn't","has","hasn","hasn't","have","haven","haven't","having","he","her","here","hers","herself","him","himself","his","how","i","if","in","into","is","isn","isn't","it","it's","its","itself","just","ll","m","ma","me","mightn","mightn't","more","most","mustn","mustn't","my","myself","needn","needn't","no","nor","not","now","o","of","off","on","once","only","or","other","our","ours","ourselves","out","over","own","re","s","same","shan","shan't","she","she's","should","should've","shouldn","shouldn't","so","some","such","t","than","that","that'll","the","their","theirs","them","themselves","then","there","these","they","this","those","through","to","too","under","until","up","ve","very","was","wasn","wasn't","we","were","weren","weren't","what","when","where","which","while","who","whom","why","will","with","won","won't","wouldn","wouldn't","y","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves","could","he'd","he'll","he's","here's","how's","i'd","i'll","i'm","i've","let's","ought","she'd","she'll","that's","there's","they'd","they'll","they're","they've","we'd","we'll","we're","we've","what's","when's","where's","who's","why's","would","able","abst","accordance","according","accordingly","across","act","actually","added","adj","affected","affecting","affects","afterwards","ah","almost","alone","along","already","also","although","always","among","amongst","announce","another","anybody","anyhow","anymore","anyone","anything","anyway","anyways","anywhere","apparently","approximately","arent","arise","around","aside","ask","asking","auth","available","away","awfully","b","back","became","become","becomes","becoming","beforehand","begin","beginning","beginnings","begins","behind","believe","beside","besides","beyond","biol","brief","briefly","current ","ca","came","cannot","can't","cause","causes","certain","certainly","co","com","come","comes","contain","containing","contains","couldnt","date","different","done","downwards","due","e","ed","edu","effect","eg","eight","eighty","either","else","elsewhere","end","ending","enough","especially","et","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","except","f","far","ff","fifth","first","five","fix","followed","following","follows","former","formerly","forth","found","four","furthermore","g","gave","get","gets","getting","give","given","gives","giving","go","goes","gone","got","gotten","h","happens","hardly","hed","hence","hereafter","hereby","herein","heres","hereupon","hes","hi","hid","hither","home","howbeit","however","hundred","id","ie","im","immediate","immediately","importance","important","inc","indeed","index","information","instead","invention","inward","itd","it'll","j","k","keep","keeps","kept","kg","km","know","known","knows","l","largely","last","lately","later","latter","latterly","least","less","lest","let","lets","like","liked","likely","line","little","'ll","look","looking","looks","ltd","made","mainly","make","makes","many","may","maybe","mean","means","meantime","meanwhile","merely","mg","might","million","miss","ml","moreover","mostly","mr","mrs","much","mug","must","n","na","name","namely","nay","nd","near","nearly","necessarily","necessary","need","needs","neither","never","nevertheless","new","next","nine","ninety","nobody","non","none","nonetheless","noone","normally","nos","noted","nothing","nowhere","obtain","obtained","obviously","often","oh","ok","okay","old","omitted","one","ones","onto","ord","others","otherwise","outside","overall","owing","p","page","pages","part","particular","particularly","past","per","perhaps","placed","please","plus","poorly","possible","possibly","potentially","pp","predominantly","present","previously","primarily","probably","promptly","proud","provides","put","q","que","quickly","quite","qv","r","ran","rather","rd","readily","really","recent","recently","ref","refs","regarding","regardless","regards","related","relatively","research","respectively","resulted","resulting","results","right","run","said","saw","say","saying","says","sec","section","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sent","seven","several","shall","shed","shes","show","showed","shown","showns","shows","significant","significantly","similar","similarly","since","six","slightly","somebody","somehow","someone","somethan","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specifically","specified","specify","specifying","still","stop","strongly","sub","substantially","successfully","sufficiently","suggest","sup","sure","take","taken","taking","tell","tends","th","thank","thanks","thanx","thats","that've","thence","thereafter","thereby","thered","therefore","therein","there'll","thereof","therere","theres","thereto","thereupon","there've","theyd","theyre","think","thou","though","thoughh","thousand","throug","throughout","thru","thus","til","tip","together","took","toward","towards","tried","tries","truly","try","trying","ts","twice","two","u","un","unfortunately","unless","unlike","unlikely","unto","upon","ups","us","use","used","useful","usefully","usefulness","uses","using","usually","v","value","various","'ve","via","viz","vol","vols","vs","w","want","wants","wasnt","way","wed","welcome","went","werent","whatever","what'll","whats","whence","whenever","whereafter","whereas","whereby","wherein","wheres","whereupon","wherever","whether","whim","whither","whod","whoever","whole","who'll","whomever","whos","whose","widely","willing","wish","within","without","wont","words","world","wouldnt","www","x","yes","yet","youd","youre","z","zero","a's","ain't","allow","allows","apart","appear","appreciate","appropriate","associated","best","better","current 'mon","current 's","cant","changes","clearly","concerning","consequently","consider","considering","corresponding","course","currently","definitely","described","despite","entirely","exactly","example","going","greetings","hello","help","hopefully","ignored","inasmuch","indicate","indicated","indicates","inner","insofar","it'd","keep","keeps","novel","presumably","reasonably","second","secondly","sensible","serious","seriously","sure","t's","third","thorough","thoroughly","three","well","wonder"}

def get_time(seconds_precision=True) -> Union[int, float]:
    """Returns the current time as Unix/Epoch timestamp, seconds precision by default"""
    return time() if not seconds_precision else int(time())


def get_uuid() -> str:
    """Returns an unique UUID (UUID4)"""
    return str(uuid4())

def keyword_extraction(sentence):
    sentence = re.sub(r'[^\w\s]', '', sentence)
    total_words = sentence.split()
    total_word_length = len(total_words)
    #print(total_word_length)
    total_sentences = tokenize.sent_tokenize(sentence)
    #print(total_sentences)
    total_sent_len = len(total_sentences)
    #print(total_sent_len)
    tf_score = {}
    for each_word in total_words:
        each_word = each_word.replace('.','')
        if each_word not in stop_words:
            if each_word in tf_score:
                tf_score[each_word] += 1
            else:
                tf_score[each_word] = 1

    # Dividing by total_word_length for each dictionary element
    tf_score.update((x, y/int(total_word_length)) for x, y in tf_score.items())
    #print(tf_score)
    def check_sent(word, sentences): 
        final = [all([w in x for w in word]) for x in sentences] 
        sent_len = [sentences[i] for i in range(0, len(final)) if final[i]]
        return int(len(sent_len))
    idf_score = {}
    for each_word in total_words:
        each_word = each_word.replace('.','')
        if each_word not in stop_words:
            if each_word in idf_score:
                idf_score[each_word] = check_sent(each_word, total_sentences)
            else:
                idf_score[each_word] = 1

    # Performing a log and divide
    idf_score.update((x, math.log(int(total_sent_len)/y)) for x, y in idf_score.items())

    #print(idf_score)
    tf_idf_score = {key: tf_score[key] * idf_score.get(key, 0) for key in tf_score.keys()}
    #print(tf_idf_score)
    keywords = tf_idf_score.keys()
    return ' '.join(keywords)

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

    matches = top_10_symptom_match(symptom, symptoms_list)
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

def aadhar_card_info(info):
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
