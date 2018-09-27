import pickle
import json
from collections import defaultdict
import os.path

topK = 10
dim = 100
# use_word = ''
use_word = '_words_second'
cate = ['Electronics', 'Cell_Phones', 'Clothing', 'CDs', 'Movies_and_TV', 'Home_and_Kitchen']
# cate = ['Clothing']
# no_features = True
no_features = False

# model = 'lightfm'
model = 'tf'

item_count = {'Electronics': 62438, 'Cell_Phones': 10329, 'Clothing': 22888, 'CDs': 64192, 'Movies_and_TV': 49880, 'Home_and_Kitchen': 28044}

def load_amazon_test(test):
    user_to_item = defaultdict(list)
    for review in test:
        user = 'u_' + review['reviewerID']
        item = 'i_' + review['asin']
        user_to_item[user].append(item)

    return user_to_item

if __name__ == '__main__':
    for c in cate:
        pred_path = '/tmp2/bschang/amazon/pred/{}_{}{}.{}.pred'.format(c, dim, use_word, model)
        test_path = '/tmp2/bschang/amazon/json/test_{}.json'.format(c)

        if no_features == True:
            pred_path = '/tmp2/bschang/amazon/pred/{}_nofeatures.{}.pred'.format(c, model)
        
        if not os.path.isfile(pred_path):
            continue
        with open(pred_path, 'rb') as f:
            pred = pickle.load(f)
        with open(test_path) as f:
            test = json.load(f)

        test_user_item = load_amazon_test(test)

        precision = []
        hit_ratio = []
        recall = []
        rec_set = set()
        for user, items in test_user_item.items():
            len_test = len(items)
            hit_count = 0
            model_pred = pred[user][:topK]
            rec_set.update(model_pred)
            # print("pred:", model_pred)
            # print("item:", items)
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
        coverage = len(rec_set) / item_count[c] * 100
        print("coverage:", "{0:.3f}%".format(coverage))

