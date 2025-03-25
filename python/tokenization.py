import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from num2words import num2words
from stopword import filterwords
# import nltk
import spacy

def is_alpha(data):
    data = data.strip()
    data = data.split(" ")
    data = [token for token in data if len(token) > 0 and token.isalpha()]
    return ' '.join(data)

def convert_lower_case(data):
    return np.char.lower(data)

def remove_stop_words(data):
    stop_words = stopwords.words("english")
    stop_words.extend(filterwords)
    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w
    return new_text

def remove_punctuation(data):
    symbols = """!\"#$%&()*+-./:;<=>?@[\\]^_`{|}~\n"""
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], " ")
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ",", "")
    return data

def remove_apostrophe(data):
    return np.char.replace(data, "'", "")

def stemming_en(data):
    stemmer = PorterStemmer()
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + stemmer.stem(w)
    return new_text

def replace_names(data, name_dict, name_counter):
    nlp_de = spacy.load("de_core_news_md")
    nlp_de.max_length = 10000000
    text = nlp_de(data)
    de_texts = ""
    for item in text.ents:
        if item.label_ == "PER":
            if item.text not in name_dict:
                mark = f'name{name_counter:04d}'
                name_counter += 1
                name_dict[item.text] = mark
            else:
                mark = name_dict[item.text]
            de_texts = de_texts + " " + mark
        else:
            de_texts = de_texts + " " + item.text

    # nlp_en = spacy.load("en_core_web_md")
    # nlp_en.max_length = 10000000
    # text = nlp_en(de_texts)
    # new_texts = ""
    # for item in text.ents:
    #     if item.label_ == "PERSON":
    #         if item.text not in name_dict:
    #             mark = f'name{name_counter:04d}'
    #             name_counter += 1
    #             name_dict[item.text] = mark
    #         else:
    #             mark = name_dict[item.text]
    #         new_texts = new_texts + " " + mark
    #     else:
    #         new_texts = new_texts + " " + item.text

    return de_texts, name_dict, name_counter


def remove_names(data):
    nlp_en = spacy.load("en_core_web_md")
    nlp_en.max_length = 10000000
    text = nlp_en(data)
    new_texts = ""
    for item in text.ents:
        if item.label_ != "PERSON":
            new_texts = new_texts + " " + item.text

    nlp_de = spacy.load("de_core_news_md")
    nlp_de.max_length = 10000000
    text = nlp_de(new_texts)
    return_texts = ""
    for item in text.ents:
        if item.label_ != "PER":
            return_texts = return_texts + " " + item.text
    return return_texts



def stemming_de(data):
    nlp = spacy.load("de_core_news_md")
    nlp.max_length = 10000000
    new_texts = ""
    for w in nlp(str(data)):
        new_texts = new_texts + " " + w.lemma_
    return new_texts

def convert_numbers(data):
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        try:
            w = num2words(int(w))
        except:
            pass
        new_text = new_text + " " + w
    new_text = np.char.replace(new_text, "-", " ")
    return new_text


# nltk.download("punkt")
# nltk.download("stopwords")
def data_clean(data, name_dict, name_counter):
    data = is_alpha(data)
    data = convert_lower_case(data)
    data = remove_punctuation(data)
    data = remove_apostrophe(data)
    data = remove_stop_words(data)
    data = convert_numbers(data)
    data = stemming_de(data)
    data = stemming_en(data)
    data, name_dict, name_counter = replace_names(data, name_dict, name_counter)
#    data = remove_names(data)
    data = remove_punctuation(data)
    data = convert_numbers(data)
    data = remove_punctuation(data)
    data = remove_stop_words(data)
    return data, name_dict, name_counter
