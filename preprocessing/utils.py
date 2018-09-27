import json
import gzip
from collections import Counter, defaultdict
from glob import glob
import ast
import networkx as nx
import nltk
from nltk.corpus import stopwords
import string
import re
from time import time

start_time = time()
punc = string.punctuation
stop = set(stopwords.words('english'))

def get_attr(obj, k):
    if k in obj:
        return obj[k]
    return ''

def not_has_num_and_alpha(s):
    return re.search('\d', s) == None

def word_filter(word):
    return (word not in punc) and (not_has_num_and_alpha(word)) and (word not in stop)

def get_price(item_type): 
    metadata_path = '/tmp2/bschang/amazon/metadata/meta_{}.json.gz'.format(item_type)
    print(metadata_path)
    
    with gzip.open(metadata_path) as f:
        for i, line in enumerate(f):
            info = {}
            line = line.decode('utf8').strip()
            meta = ast.literal_eval(line)
            item = 'i_' + get_attr(meta, 'asin')
            price = get_attr(meta, 'price')

            print(item, price)


get_price('Cell_Phones')
