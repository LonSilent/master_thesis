import numpy as np
from scipy.sparse import csr_matrix, identity
from lightfm import LightFM
from timeit import default_timer
from time import time
import pickle
import os
import json
from collections import defaultdict
from gensim.models import KeyedVectors

# const
dim = 100
threads = 12
step = 1000
topK = 30
epoch = 150
K = 150

# use_word = ''
use_word = '_words'
# use_word = '_words_ids'
# use_ids = True

# no_features = False
no_features = True

loss_function = 'bpr'
# loss_function = 'warp'

# c = 'Electronics'
# c = 'Cell_Phones'
# c = 'Clothing'
c = 'CDs'
# c = 'Movies_and_TV'
# c = 'Home_and_Kitchen' 
# c = 'jd'

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

def construct_features(array_set, embedding):
    row = []
    column = []
    data = []
    length = len(array_set)

    for index, item in enumerate(array_set):
        vector = embedding[item]
        row.append(index)
        column.append(index)
        data.append(1.0)
        for j, elem in enumerate(vector):
            row.append(index)
            column.append(length + j)
            data.append(elem)

    row = np.asarray(row)
    column = np.asarray(column)
    data = np.asarray(data)
    features = csr_matrix((data, (row, column)), shape=(len(array_set), len(array_set) + dim))

    # print(features[100])
    return features

if __name__ == '__main__':
    train_path = '/tmp2/bschang/amazon/json/train_{}.json'.format(c)
    embd_path = '/tmp2/bschang/amazon/embd/{}_{}{}.embd.txt'.format(c, dim, use_word)
    pred_path = '/tmp2/bschang/amazon/pred/{}_{}{}.lightfm.pred'.format(c, dim, use_word)
    
    if c == 'jd':
        train_path = '/tmp2/bschang/jd/new_train_5.json'
        embd_path = '/tmp2/bschang/jd/embd/jd_{}{}.embd.txt'.format(dim, use_word)
        pred_path = '/tmp2/bschang/jd/pred/jd_{}{}.lightfm.pred'.format(dim, use_word)

    print('embd path:', embd_path)
    print('pred path:', pred_path)

    with open(train_path) as f:
        train = json.load(f)
    # print(train[:2])
    if 'amazon' in train_path:
        user_to_item_score = load_amazon_file(train)
    elif 'jd' in train_path:
        user_to_item_score = load_jd_file(train)
    
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
    print(item_set[:2])
    user_index = {v: i for i, v in enumerate(user_set)}
    item_index = {v: i for i, v in enumerate(item_set)} 

    ui_matrix = dict_to_sparse(user_to_item_score, user_index, item_index, user_set, item_set)
    print("interactions shape:", ui_matrix.shape)
    print(ui_matrix[0])

    print("Loading embedding...")
    embedding = KeyedVectors.load_word2vec_format(embd_path)
    
    user_features = np.zeros((len(user_set), dim))
    for index, user in enumerate(user_set):
        user_features[index, :] = embedding[user]
    item_features = np.zeros((len(item_set), dim))
    for index, item in enumerate(item_set):
        item_features[index, :] = embedding[item]
    user_features = csr_matrix(user_features)
    # user_features = identity(len(user_set), format='csr')
    item_features = csr_matrix(item_features)
    # item_features = identity(len(item_set), format='csr')
    
    # user_features = construct_features(user_set, embedding)
    # item_features = construct_features(item_set, embedding)
    print("user features shape:", user_features.shape)
    print("item features shape:", item_features.shape)

    if no_features:
        user_features = identity(len(user_set), format='csr')
        item_features = identity(len(item_set), format='csr')
        if 'amazon' in pred_path:
            pred_path = '/tmp2/bschang/amazon/pred/{}_nofeatures.lightfm.pred'.format(c)
        else:
            pred_path = '/tmp2/bschang/jd/pred/jd_nofeatures.lightfm.pred'
        print("Apply pure mf")
        print("user features shape:", user_features.shape)
        print("item features shape:", item_features.shape)

    start = time()
    print("Start training lightfm...")
    print("param:")
    print("no_components: {}, loss: {}, epochs: {}, no_features: {}". format(
        K, loss_function, epoch, no_features))
    model = LightFM(no_components=K, loss=loss_function)
    model.fit(ui_matrix,
            user_features=user_features,
            item_features=item_features,
            epochs=epoch,
            num_threads=threads)
    end = time()
    print("Spend {0:.2f} secs for training lightFM.".format(end - start))

    users = list(range(user_features.shape[0]))
    recommendation = {}
    print("Start predicting...")
    start = time()
    item_id = np.array(range(len(item_set)))
    user_counter = 0
    batch_counter = 0
    for i in range(0, user_features.shape[0], step):
        batch_start = time()
        upper_bound = min(i + step, user_features.shape[0])
        # print("processing batch {}...".format(batch_counter))
        batch = np.array(users[i:upper_bound])
        batch_size = batch.shape[0]
        user_ids = np.array(batch)
        user_ids = np.repeat(user_ids, len(item_set), axis=0)
        item_ids = np.tile(item_id, batch_size)
        pred = model.predict(user_ids, item_ids,
                item_features=item_features,
                user_features=user_features,
                num_threads=threads)
        pred = pred.reshape(batch_size, len(item_set))
        for p in pred:
            top_items = top_result_ids(item_set, p, topK)
            user_id = user_set[user_counter]
            top_items = [x for x in top_items if x not in train_items[user_id]]
            recommendation[user_id] = top_items
            user_counter += 1
        batch_end = time()
        print("Finish batch {}, {}~{} users".format(batch_counter, i, upper_bound), 
                "cost {0:.2f} secs".format(batch_end-batch_start), end='\r')
        batch_counter += 1
    end = time()
    print()
    print("Cost {0:.2f} secs for prediction".format(end - start))

    with open(pred_path, 'wb') as f:
        pickle.dump(recommendation, f) 

    pred = recommendation
    if 'amazon' in train_path:
        test_path = '/tmp2/bschang/amazon/json/test_{}.json'.format(c)
    elif 'jd' in train_path:
        test_path = '/tmp2/bschang/jd/test_5.json'
    with open(test_path) as f:
        test = json.load(f)

    topK = 10
    if 'amazon' in train_path:
        test_user_item = load_amazon_test(test)
    elif 'jd' in train_path:
        test_user_item = load_jd_test(test)

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

    print(c)
    avg_p = sum(precision) / len(precision) * 100
    print("avg precision at", topK, ": {0:.3f}%".format(avg_p))
    avg_hit_ratio = sum(hit_ratio) / len(hit_ratio) * 100
    print("hit raio:", "{0:.3f}%".format(avg_hit_ratio))
    avg_recall = sum(recall) / len(recall) * 100
    print("avg recall:", "{0:.3f}%".format(avg_recall))
    coverage = len(rec_items) / item_count * 100
    print("coverage:", "{0:.3f}%".format(coverage))

