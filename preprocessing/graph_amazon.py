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
import sys

start_time = time()
punc = string.punctuation
# print(punc)
stop = set(stopwords.words('english'))
# print(stop)
# categories = ['Electronics']
categories = ['Electronics', 'CDs', 'Clothing', 'Cell_Phones', 'Movies_and_TV', 'Home_and_Kitchen']
# categories = ['Electronics', 'CDs', 'Clothing', 'Cell_Phones']
# categories = ['Movies_and_TV', 'Home_and_Kitchen']

def get_attr(obj, k):
    if k in obj:
        return obj[k]
    return ''

def not_has_num_and_alpha(s):
    # number = re.search('\d', s)
    return re.search('\d', s) == None

def word_filter(word):
    return (word not in punc) and (not_has_num_and_alpha(word)) and (word not in stop)

# for item_type in categories:
#     users = set()
#     items = set()
#     rating_count = 0
#     user_to_item = defaultdict(list)

#     user2item_edgelist = []
#     # data_path = '/tmp2/bschang/amazon/json/reviews_{}.json.gz'.format(item_type)
#     data_path = '/tmp2/bschang/amazon/json/train_{}.json'.format(item_type)
#     print(data_path)
#     # with gzip.open(data_path) as f:
#         # for line in f:
#             # line = line.decode('utf8').strip().split('\t')
#             # review = json.loads(line[0])
#             # user = 'u_' + review['reviewerID']
#             # item = 'i_' + review['asin']
#             # rating = float(review['overall'])
#             # rating_count += 1
#             # users.add(user)
#             # items.add(item)
#             # user_to_item[user].append(item)
#             # user2item_edgelist.append((user, item, rating))
#     with open(data_path) as f:
#         train = json.load(f)
#     for review in train:
#         # print(i)
#         user = 'u_' + review['reviewerID']
#         item = 'i_' + review['asin']
#         rating = float(review['overall'])
#         rating_count += 1
#         users.add(user)
#         items.add(item)
#         user_to_item[user].append(item)
#         user2item_edgelist.append((user, item, rating))

#     print('{} users, {} items, {} raitngs.'.format(len(users), len(items), rating_count))
#     sum_of_item = sum([len(v) for v in user_to_item.values()])
#     avg_item = sum_of_item / len(user_to_item.keys())
#     print('user avg items:', avg_item)
    
#     metadata_path = '/tmp2/bschang/amazon/metadata/meta_{}.json.gz'.format(item_type)
#     print(metadata_path)
    
#     item2word_edgelist = []
#     item2cate_edgelist = []
#     total_words = set()
#     total_cates = set()
#     total_items = set()
#     with gzip.open(metadata_path) as f:
#         for i, line in enumerate(f):
#             info = {}
#             line = line.decode('utf8').strip()
#             meta = ast.literal_eval(line)
#             # print(meta)
#             # items.add(meta['asin'])
#             item = 'i_' + get_attr(meta, 'asin')
#             if item not in items:
#                 continue
#             title = get_attr(meta, 'title')
#             words = set([word.lower() for word in nltk.word_tokenize(title) if word_filter(word)])
#             cate = get_attr(meta, 'categories')[0]
#             total_words.update(words)
#             total_cates.update(cate)
#             brand = get_attr(meta, 'brand')
#             total_items.add(item)
#             if brand != '':
#                 words.add(brand.lower().replace(' ', '_'))
#                 total_words.add(brand)
#             for w in words:
#                 item2word_edgelist.append((item, 'w_' + w, 1.0))
#             for c in cate:
#                 item2cate_edgelist.append((item, 'c_' + c, 1.0))
#             print('{}'.format(i), end='\r')
#             # print(title)
#         print('{} items, {} words, {} cates'.format(len(total_items), len(total_words), len(total_cates)))


#     print('Write edgelist...')
#     edgelist = []
#     graph = nx.Graph()
#     result_path = '/tmp2/bschang/amazon/network/train_network_{}.pickle'.format(item_type)
#     for edge in user2item_edgelist:
#         graph.add_edge(edge[0], edge[1], weight=edge[2])
#     for edge in item2cate_edgelist:
#         graph.add_edge(edge[0], edge[1], weight=edge[2])
#     for edge in item2word_edgelist:
#         graph.add_edge(edge[0], edge[1], weight=edge[2])

#     nx.write_gpickle(graph, result_path)

# end_time = time()
# print("Finished and spend {0:.2f} secs".format(end_time - start_time))

data_path = sys.argv[1]
metadata_path = sys.argv[2]
result_path = sys.argv[3]

users = set()
items = set()
rating_count = 0
user_to_item = defaultdict(list)

user2item_edgelist = []
# data_path = '/tmp2/bschang/amazon/json/reviews_{}.json.gz'.format(item_type)
# data_path = '/tmp2/bschang/amazon/json/train_{}.json'.format(item_type)
print(data_path)
# with gzip.open(data_path) as f:
    # for line in f:
        # line = line.decode('utf8').strip().split('\t')
        # review = json.loads(line[0])
        # user = 'u_' + review['reviewerID']
        # item = 'i_' + review['asin']
        # rating = float(review['overall'])
        # rating_count += 1
        # users.add(user)
        # items.add(item)
        # user_to_item[user].append(item)
        # user2item_edgelist.append((user, item, rating))
with open(data_path) as f:
    train = json.load(f)
for review in train:
    # print(i)
    user = 'u_' + review['reviewerID']
    item = 'i_' + review['asin']
    rating = float(review['overall'])
    rating_count += 1
    users.add(user)
    items.add(item)
    user_to_item[user].append(item)
    user2item_edgelist.append((user, item, rating))

print('{} users, {} items, {} raitngs.'.format(len(users), len(items), rating_count))
sum_of_item = sum([len(v) for v in user_to_item.values()])
avg_item = sum_of_item / len(user_to_item.keys())
print('user avg items:', avg_item)

# metadata_path = '/tmp2/bschang/amazon/metadata/meta_{}.json.gz'.format(item_type)
print(metadata_path)

item2word_edgelist = []
item2cate_edgelist = []
total_words = set()
total_cates = set()
total_items = set()
with gzip.open(metadata_path) as f:
    for i, line in enumerate(f):
        info = {}
        line = line.decode('utf8').strip()
        meta = ast.literal_eval(line)
        # print(meta)
        # items.add(meta['asin'])
        item = 'i_' + get_attr(meta, 'asin')
        if item not in items:
            continue
        title = get_attr(meta, 'title')
        words = set([word.lower() for word in nltk.word_tokenize(title) if word_filter(word)])
        cate = get_attr(meta, 'categories')[0]
        total_words.update(words)
        total_cates.update(cate)
        brand = get_attr(meta, 'brand')
        total_items.add(item)
        if brand != '':
            words.add(brand.lower().replace(' ', '_'))
            total_words.add(brand)
        for w in words:
            item2word_edgelist.append((item, 'w_' + w, 1.0))
        for c in cate:
            item2cate_edgelist.append((item, 'c_' + c, 1.0))
        print('{}'.format(i), end='\r')
        # print(title)
    print('{} items, {} words, {} cates'.format(len(total_items), len(total_words), len(total_cates)))


print('Write edgelist...')
edgelist = []
graph = nx.Graph()
# result_path = '/tmp2/bschang/amazon/network/train_network_{}.pickle'.format(item_type)
for edge in user2item_edgelist:
    graph.add_edge(edge[0], edge[1], weight=edge[2])
for edge in item2cate_edgelist:
    graph.add_edge(edge[0], edge[1], weight=edge[2])
for edge in item2word_edgelist:
    graph.add_edge(edge[0], edge[1], weight=edge[2])

nx.write_gpickle(graph, result_path)

end_time = time()
print("Finished and spend {0:.2f} secs".format(end_time - start_time))

