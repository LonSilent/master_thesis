import json, gzip
import ast
from gensim.models import KeyedVectors
import random
import sys
import pickle, json
import os
from collections import defaultdict

def get_attr(obj, k):
    if k in obj:
        return obj[k]
    return ''

def print_items(l, item_title):
    count = 0
    for item, sim in l:
        if item.startswith('i_') and item_title[item] != '':
            print(item, item_title[item])
            count += 1
        if count > 4:
            break

def load_metadata(c):
    item_info = {}
    metadata_path = '/tmp2/bschang/amazon/metadata/meta_{}.json.gz'.format(c)
    with gzip.open(metadata_path) as f:
        items = set()
        for i, line in enumerate(f):
            info = {}
            line = line.decode('utf8').strip()
            meta = ast.literal_eval(line)
            items.add(meta['asin'])
            item = 'i_' + get_attr(meta, 'asin')
            title = get_attr(meta, 'title')
            cate = get_attr(meta, 'categories')[0]
            brand = get_attr(meta, 'brand')
            item_info[item] = {'title': title, 'cate': cate}
            print('{}'.format(i), end='\r')
        print('items count:', len(items))

    return item_info

def load_amazon_train(path):
    users = set()
    user_to_item = defaultdict(list)
    with open(path) as f:
        train = json.load(f)
    for i in train:
        user = 'u_' + i['reviewerID']
        item = 'i_' + i['asin']
        user_to_item[user].append(item)
        users.add(user)

    return list(users), user_to_item

def load_amazon_test(path):
    user_to_item = defaultdict(list)
    with open(path) as f:
        test = json.load(f)
    for i in test:
        user = 'u_' + i['reviewerID']
        item = 'i_' + i['asin']
        user_to_item[user].append(item)

    return user_to_item

def parse_title(meta):
    item_title = {}
    for item, value in meta.items():
        item_title[item] = value['title']
    
    with open('/tmp2/bschang/amazon/metadata/{}_title.pickle'.format(c), 'wb') as f:
        pickle.dump(item_title, f)

    return item_title

def search_title(query, item_title):
    hit_list = []
    for item, value in item_title.items():
        if query in value.lower():
            hit_list.append(value)

    print(hit_list)


c = sys.argv[1]
train_path = '/tmp2/bschang/amazon/json/train_{}.json'.format(c)
test_path = '/tmp2/bschang/amazon/json/test_{}.json'.format(c)
title_path = '/tmp2/bschang/amazon/metadata/{}_title.pickle'.format(c)

if os.path.isfile(title_path):
    with open(title_path, 'rb') as f:
        item_title = pickle.load(f)
else:    
    metadata = load_metadata(c)
    item_title = parse_title(metadata)
# print(len(item_title.keys()))
print(list(item_title.values())[:10])
# exit()

users, train = load_amazon_train(train_path)
test = load_amazon_test(test_path)
item_id = 'i_B00FGOTBQO' # Electronics
embd_path = {'metapath': '/tmp2/bschang/amazon/embd/{}_100_words.embd.txt'.format(c),
        'deepwalk': '/tmp2/bschang/amazon/embd/{}_100_deepwalk.embd.txt'.format(c),
        'hpe': '/tmp2/bschang/amazon/embd/{}_100_hpe.embd.txt'.format(c),
        'line': '/tmp2/bschang/amazon/embd/{}_100_line.embd.txt'.format(c)}
pred_path = {'metapath': '/tmp2/bschang/amazon/pred/{}_100_words.tf.pred'.format(c),
        'deepwalk': '/tmp2/bschang/amazon/pred/{}_100_deepwalk.tf.pred'.format(c),
        'hpe': '/tmp2/bschang/amazon/pred/{}_100_hpe.tf.pred'.format(c),
        'line': '/tmp2/bschang/amazon/pred/{}_100_line.tf.pred'.format(c)}
# print("item:", item_id, item_title[item_id])
# for method, path in embd_path.items():
    # vectors = KeyedVectors.load_word2vec_format(path, binary=False)
    # print("method:", method)
    # print_items(vectors.similar_by_word(item_id, topn=30), item_title)
    # print('============================')

user_id = 'u_A2WW1SJMFUYMQT'
test_item = test[user_id][0]
print("user:", user_id)
print("train items:")
for item in train[user_id]:
    print(item, item_title[item])
print("test items:")
print(test_item, item_title[test_item])
print("=====================")
for method, path in pred_path.items():
    counter = 0
    with open(path, 'rb') as f:
        pred = pickle.load(f)
    print(method, "pred:")
    if test_item in pred[user_id][:10]:
        print("HIIIIIIIIIT!!!!")
    for item in pred[user_id][:10]:
        if item_title[item] != '':
            print(item, item_title[item])
            counter += 1
        if counter > 4:
            break

