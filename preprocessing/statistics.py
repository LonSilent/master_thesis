import json, gzip
from collections import defaultdict


categories = ['Electronics', 'CDs', 'Cell_Phones', 'Clothing', 'Movies_and_TV', 'Home_and_Kitchen']

for item_type in categories:
    data_path = '/tmp2/bschang/amazon/json/reviews_{}.json.gz'.format(item_type)
    print(data_path)

    users = set()
    items = set()
    user_to_item = defaultdict(list)
    rating = 0
    with gzip.open(data_path) as f:
        for line in f:
            line = line.decode('utf8').strip().split('\t')
            review = json.loads(line[0])
            # print(review)
            user = 'u_' + review['reviewerID']
            item = 'i_' + review['asin']
            user_to_item[user].append(item)
            users.add(user)
            items.add(item)
            rating += 1

    user_count = len(users)
    item_count = len(items)
    sparsity = rating / (user_count * item_count) * 100
    avg_item = sum([len(v) for v in user_to_item.values()]) / user_count
    print("user count: {}, item_count: {}, rating count: {}".format(len(users), len(items), rating))
    print("avg item: {}, sparsity: {}%".format(avg_item, sparsity))
