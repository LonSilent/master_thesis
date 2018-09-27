import json, gzip, sys
from collections import defaultdict

def train_test_split(instance, train_percentage=0.8):
    train_len = int(len(instance) * train_percentage)
    return instance[:train_len], instance[train_len:]

categories = ['Electronics', 'CDs', 'Cell_Phones', 'Clothing']
categories = ['Movies_and_TV', 'Home_and_Kitchen']

# for item_type in categories:
#     data_path = '/tmp2/bschang/amazon/json/reviews_{}.json.gz'.format(item_type)
#     train_path = '/tmp2/bschang/amazon/json/train_{}.json'.format(item_type)
#     test_path = '/tmp2/bschang/amazon/json/test_{}.json'.format(item_type)
#     print(data_path)

#     users = set()
#     items = set()
#     user_to_item = defaultdict(list)
#     user2item_edgelist = []
#     train = []
#     test = []
#     with gzip.open(data_path) as f:
#         for line in f:
#             line = line.decode('utf8').strip().split('\t')
#             review = json.loads(line[0])
#             # print(review)
#             user = 'u_' + review['reviewerID']
#             item = 'i_' + review['asin']
#             rating = float(review['overall'])
#             users.add(user)
#             items.add(item)
#             user_to_item[user].append(review)
#             # user2item_edgelist.append((user, item, rating))
    
#     # test_user = list(user_to_item.keys())[0]
#     # print(user_to_item[test_user])
#     # print(sorted(user_to_item[test_user], key=lambda x:x['unixReviewTime']))

#     for user, history in user_to_item.items():
#         sort_items = sorted(user_to_item[user], key=lambda x:x['unixReviewTime'])
#         train_items, test_items = train_test_split(sort_items, train_percentage=0.8)
#         # print("train:", train_items)
#         # print("test:", test_items)
#         for i in train_items:
#             train.append(i)
#         for i in test_items:
#             test.append(i)

#     with open(train_path, 'w') as f:
#         json.dump(train, f)
#     with open(test_path, 'w') as f:
#         json.dump(test, f)

# categories = ['Electronics', 'CDs', 'Cell_Phones', 'Clothing']
# categories = ['Movies_and_TV', 'Home_and_Kitchen']

# for item_type in categories:
data_path = sys.argv[1]
train_path = sys.argv[2]
test_path = sys.argv[3]
# data_path = '/tmp2/bschang/amazon/json/reviews_{}.json.gz'.format(item_type)
# train_path = '/tmp2/bschang/amazon/json/train_{}.json'.format(item_type)
# test_path = '/tmp2/bschang/amazon/json/test_{}.json'.format(item_type)
print(data_path)

users = set()
items = set()
user_to_item = defaultdict(list)
user2item_edgelist = []
train = []
test = []
with gzip.open(data_path) as f:
    for line in f:
        line = line.decode('utf8').strip().split('\t')
        review = json.loads(line[0])
        # print(review)
        user = 'u_' + review['reviewerID']
        item = 'i_' + review['asin']
        rating = float(review['overall'])
        users.add(user)
        items.add(item)
        user_to_item[user].append(review)
        # user2item_edgelist.append((user, item, rating))

# test_user = list(user_to_item.keys())[0]
# print(user_to_item[test_user])
# print(sorted(user_to_item[test_user], key=lambda x:x['unixReviewTime']))

for user, history in user_to_item.items():
    sort_items = sorted(user_to_item[user], key=lambda x:x['unixReviewTime'])
    train_items, test_items = train_test_split(sort_items, train_percentage=0.8)
    # print("train:", train_items)
    # print("test:", test_items)
    for i in train_items:
        train.append(i)
    for i in test_items:
        test.append(i)

with open(train_path, 'w') as f:
    json.dump(train, f)
with open(test_path, 'w') as f:
    json.dump(test, f)
