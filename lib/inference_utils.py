import re
import pandas as pd

from razdel import sentenize

import torch
import torch.nn.functional as F







def clean(string):    
    string = re.sub('[^а-яА-Яa-zA-Z0-9)(+-/@.,# \n]', '', string)
    string = re.sub('\)+', ')', string)
    string = re.sub('\(+', '(', string)
    string = re.sub('\++', '+', string)
    string = re.sub('\-+', '-', string)
    string = re.sub('\/+', '/', string)
    string = re.sub('\@+', '@', string)
    string = re.sub('\.+', '.', string)
    string = re.sub('\,+', ',', string)
    string = re.sub('\#+', '#', string)
    string = re.sub('\ +', ' ', string)
    string = re.sub('/\n', '\n', string)
    string = re.sub('\n+', '\n', string)
    
    return string


def to_sents(text):    
    paragraphs = [p for p in text.split('\n')]
    full_list = []
    for paragraph in paragraphs:
        sents = list(sentenize(paragraph))
        full_list.append(sents)
    full_list = [sent for sents in full_list for sent in sents if sent]
    full_list = [sent.text for sent in full_list if sent.text]
    
    return full_list


def embed(x, tokenizer, class_model):    
    if len(x) == 0:
        x = 'Пусто' 
        
    tokenized_x = tokenizer(x, padding = True,
                             truncation = True,
                             max_length = 512,
                             return_tensors='pt')
    
    with torch.no_grad():      
        model_output = class_model(**tokenized_x)
    
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = F.normalize(embeddings)
    
    return embeddings


def classify(entry, lstm_classifier_head):   
    with torch.no_grad():
        x = torch.exp(lstm_classifier_head(entry.unsqueeze(0))).numpy()[:, 1]
        
    return x


def perform_ner(entry, entry_razdel, ner_model):
    is_found = False
    phrase_len_thresh = 5
    
    for sentence in entry_razdel:
        ner_output = ner_model(sentence)
        if len(ner_output) == 0:
            continue
            
        phrases = list(
            filter(
                lambda x: len(x) > phrase_len_thresh, [x['word'] for x in ner_output]
            ))    
        num_found = len(phrases)
        
        if num_found == 0:
            continue
            
        elif num_found > 1:
            return [None, None]
        
        else:
            phrase = phrases[0]
            if is_found == True:
                return [None, None]
            clean_chars = [char for char in phrase if char.isalnum()]
            re_string = '.{0,9}'.join(clean_chars[:-1])
            is_found = True
    if not is_found:
        return [None, None]  
    try:
        match = re.search(re_string, entry)
        span, word = match.span(0), match.group(0)
        return [span[0], span[1]]
    except:
        return [None, None]