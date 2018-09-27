import numpy as np
from scipy.sparse import csr_matrix, identity
# from lightfm import LightFM
from timeit import default_timer
from time import time
import pickle
import os
import json
from collections import defaultdict
from gensim.models import KeyedVectors
import networkx as nx
import sys

sys.path.append('.')

import tensorrec
from tensorrec.loss_graphs import WMRBLossGraph, BalancedWMRBLossGraph, BPRLossGraph
from tensorrec.representation_graphs import ReLURepresentationGraph, NormalizedLinearRepresentationGraph
from tensorrec.prediction_graphs import DotProductPredictionGraph, CosineSimilarityPredictionGraph, EuclidianSimilarityPredictionGraph

print("import")


# const
dim = 100
topK = 30
epoch = 150
K = 150
n_sample = 20
user_batch_size = None
# user_batch_size = 50000
# user_batch_size = 65000
shuffle_batch = True
# shuffle_batch = False

# use_word = ''
# use_word = '_words'
# use_word = '_words_second'
# use_word = '_words_third'
# use_word = '_deepwalk'
# use_word = '_deepwalk_noword'
# use_word = '_deepwalk_pronet'
# use_word = '_line'
# use_word = '_line_noword'
# use_word = '_hpe'
# use_word = '_hpe_noword'

# use_text = 'text'
# use_text = 'text_second'
# use_text = 'text_third'
use_text = False

no_features = False
# no_features = True

# use_bag_of_word = True
use_bag_of_word = False

# c = 'Electronics'
# c = 'Cell_Phones'
# c = 'Clothing'
# c = 'CDs'
# c = 'Movies_and_TV'
# c = 'Home_and_Kitchen'
# c = 'jd'

method = 'ranking'
# method = 'mf'


def load_amazon_test(test):
    user_to_item = defaultdict(list)
    for review in test:
        user = 'u_' + review['reviewerID']
        item = 'i_' + review['asin']
        user_to_item[user].append(item)

    return user_to_item

def load_jd_test(test):
    user_to_item = defaultdict(list)
    for review in test:
        user = review['user']
        item = review['item']
        user_to_item[user].append(item)

    return user_to_item

def get_neighbors(node):
    return list(node.keys())

def top_result_ids(index_item, pred, topK):
    top_ids = []
    top_index = pred.argsort()[-topK:][::-1]
    for item_idx in top_index:
        top_ids.append(index_item[item_idx])

    return top_ids


def dict_to_sparse(obj, user_index, item_index, user_set, item_set):
    row = []
    column = []
    data = []
    # print(obj)
    for user, items in obj.items():
        for item, score in items.items():
            row.append(user_index[user])
            column.append(item_index[item])
            data.append(score)
    # print(row[:10], column[:10], data[:10])

    row = np.asarray(row)
    column = np.asarray(column)
    data = np.asarray(data)
    sparse_matrix = csr_matrix((data, (row, column)), shape=(len(user_set), len(item_set)))

    return sparse_matrix


def load_amazon_file(data):
    user_to_item_score = defaultdict(dict)
    for i in data:
        user = 'u_' + i['reviewerID']
        item = 'i_' + i['asin']
        rating = float(i['overall'])
        # if int(rating) >= 4:
        # rating = 1.0
        # else:
        # continue
        user_to_item_score[user][item] = rating

    return user_to_item_score


def load_jd_file(data):
    user_to_item_score = defaultdict(dict)
    for i in data:
        user = i['user']
        item = i['item']
        rating = float(i['score'])
        user_to_item_score[user][item] = rating

    return user_to_item_score


def construct_item_features(item_set, item_words, word_set):
    all_features = []
    all_features.extend(word_set)
    # all_features.extend(item_set)

    row = []
    column = []
    data = []

    counter = 0
    index_of_features = {x: i for i, x in enumerate(all_features)}
    for item in item_set:
        # row.append(counter)
        # column.append(index_of_features[item])
        # data.append(1.0)
        for word in item_words[item]:
            row.append(counter)
            column.append(index_of_features[word])
            data.append(1.0)
        counter += 1

    row = np.asarray(row)
    column = np.asarray(column)
    data = np.asarray(data)

    features = csr_matrix((data, (row, column)), shape=(counter, len(index_of_features.keys())))
    return features


if __name__ == '__main__':
    # train_path = '/tmp2/bschang/amazon/json/train_{}.json'.format(c)
    # embd_path = '/tmp2/bschang/amazon/embd/{}_{}{}.embd.txt'.format(c, dim, use_word)
    # pred_path = '/tmp2/bschang/amazon/pred/{}_{}{}.tf.pred'.format(c, dim, use_word)

    # if c == 'jd':
    #     train_path = '/tmp2/bschang/jd/new_train_5.json'
    #     embd_path = '/tmp2/bschang/jd/embd/jd_{}{}.embd.txt'.format(dim, use_word)
    #     pred_path = '/tmp2/bschang/jd/pred/jd_{}{}.tf.pred'.format(dim, use_word)

    train_path = sys.argv[1]
    test_path = sys.argv[2]
    embd_path = sys.argv[3]
    pred_path = sys.argv[4]

    print("train path:", train_path)
    print("test path:", test_path)
    print('embd path:', embd_path)
    print('pred path:', pred_path)

    with open(train_path) as f:
        train = json.load(f)
    user_to_item_score = load_amazon_file(train)

    train_items = defaultdict(set)
    user_set = set()
    item_set = set()
    for user, items in user_to_item_score.items():
        user_set.add(user)
        for item in items:
            train_items[user].add(item)
            item_set.add(item)

    print("{} users and {} items".format(len(user_set), len(item_set)))
    user_set = list(user_set)
    item_set = list(item_set)
    item_count = len(item_set)
    user_index = {v: i for i, v in enumerate(user_set)}
    item_index = {v: i for i, v in enumerate(item_set)}

    ui_matrix = dict_to_sparse(user_to_item_score, user_index, item_index, user_set, item_set)
    print("interactions shape:", ui_matrix.shape)

    print("Loading embedding...")
    embedding = KeyedVectors.load_word2vec_format(embd_path)
    user_features = np.zeros((len(user_set), dim))
    for index, user in enumerate(user_set):
        user_features[index, :] = embedding[user]
    item_features = np.zeros((len(item_set), dim))
    for index, item in enumerate(item_set):
        item_features[index, :] = embedding[item]

    user_features = csr_matrix(user_features)
    item_features = csr_matrix(item_features)
    print("user features shape:", user_features.shape)
    print("item features shape:", item_features.shape)

    if no_features:
        user_features = identity(len(user_set), format='csr')
        item_features = identity(len(item_set), format='csr')
        pred_path = pred_path + '.no_features'
        # if 'amazon' in pred_path:
        #     pred_path = '/tmp2/bschang/amazon/pred/{}_nofeatures.tf.pred'.format(c)
        # else:
        #     pred_path = '/tmp2/bschang/jd/pred/jd_nofeatures.tf.pred'
        print("Apply pure mf")
        print("user features shape:", user_features.shape)
        print("item features shape:", item_features.shape)

    if use_bag_of_word:
        print("use bag of words")
        user_features = identity(len(user_set), format='csr')
        network_path = '/tmp2/bschang/amazon/network/train_network_{}.pickle'.format(c)
        graph = nx.read_gpickle(network_path)
        word_set = set()
        item_words = {}
        for item in item_set:
            words = [x for x in get_neighbors(graph[item]) if x.startswith('w_')]
            item_words[item] = words
            word_set.update(words)
        word_set = list(word_set)
        word_index = {v: i for i, v in enumerate(word_set)}
        print("{} words".format(len(word_set)))
        item_features = construct_item_features(item_set, item_words, word_set)
        print("user features shape:", user_features.shape)
        print("item features shape:", item_features.shape)
        pred_path = '/tmp2/bschang/amazon/pred/{}_bow.tf.pred'.format(c)

    start = time()
    print("Start training tensorrec...")
    print("params")
    print("n_components: {}, epochs: {}, n_sample: {}, user_batch_size: {}, shuffle_batch: {}, method: {}".format(
        K, epoch, n_sample, user_batch_size, shuffle_batch, method))
    if user_batch_size is None:
        user_batch_size = len(user_set)
    if method == 'ranking':
        model = tensorrec.TensorRec(
            n_components=K,
            # user_repr_graph=NormalizedLinearRepresentationGraph(),
            item_repr_graph=NormalizedLinearRepresentationGraph(),
            loss_graph=BalancedWMRBLossGraph(),
            # loss_graph=BPRLossGraph(),
            normalize_users=True,
            normalize_items=True
        )
    elif method == 'mf':
        model = tensorrec.TensorRec(
            n_components=K,
        )
    model.fit(ui_matrix, user_features, item_features,
              epochs=epoch,
              verbose=False,
              n_sampled_items=n_sample,
              user_batch_size=user_batch_size,
              shuffle_batch=shuffle_batch
              )
    end = time()
    print("Spend {0:.2f} secs for training tensorrec.".format(end - start))

    recommendation = {}
    print("Start predicting...")
    start = time()
    pred = model.predict(user_features=user_features, item_features=item_features)
    for index, p in enumerate(pred):
        user_id = user_set[index]
        top_items = top_result_ids(item_set, p, topK)
        top_items = [x for x in top_items if x not in train_items[user_id]]
        recommendation[user_id] = top_items
    end = time()
    print("Cost {0:.2f} secs for prediction".format(end - start))

    with open(pred_path, 'wb') as f:
        pickle.dump(recommendation, f)

    pred = recommendation
    # with open(pred_path, 'rb') as f:
    #     pred = pickle.load(f)
    # if 'amazon' in train_path:
    #     test_path = '/tmp2/bschang/amazon/json/test_{}.json'.format(c)
    # elif 'jd' in train_path:
    #     test_path = '/tmp2/bschang/jd/test_5.json'
    with open(test_path) as f:
        test = json.load(f)

    topK = 10
    test_user_item = load_amazon_test(test)

    precision = []
    hit_ratio = []
    recall = []
    rec_items = set()
    for user, items in test_user_item.items():
        len_test = len(items)
        hit_count = 0
        model_pred = pred[user][:topK]
        rec_items.update(model_pred)
        for item in items:
            if item in model_pred:
                # print("hit")
                hit_count += 1
        # precision
        p = hit_count / topK
        precision.append(p)
        # hit ratio
        if hit_count > 0:
            hit_ratio.append(1)
        else:
            hit_ratio.append(0)
        # recall
        r = hit_count / len_test
        recall.append(r)

    # print(c)
    avg_p = sum(precision) / len(precision) * 100
    print("avg precision at", topK, ": {0:.3f}%".format(avg_p))
    avg_hit_ratio = sum(hit_ratio) / len(hit_ratio) * 100
    print("hit raio:", "{0:.3f}%".format(avg_hit_ratio))
    avg_recall = sum(recall) / len(recall) * 100
    print("avg recall:", "{0:.3f}%".format(avg_recall))
    coverage = len(rec_items) / item_count * 100
    print("coverage:", "{0:.3f}%".format(coverage))
