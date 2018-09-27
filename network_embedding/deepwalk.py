import networkx as nx
import pickle
import random
from time import time
import argparse
from gensim.models.word2vec import Word2Vec

# constant
USER_TYPE = 0
ITEM_TYPE = 1
WORD_TYPE = 2
CATEGORY_TYPE = 3
WALK_TIMES = 10
WALK_LENGTH = 50

def get_neighbors(node):
    return list(node.keys())

def get_type(s):
    if s.startswith('u'):
        return USER_TYPE
    elif s.startswith('i'):
        return ITEM_TYPE
    elif s.startswith('w'):
        return WORD_TYPE
    elif s.startswith('c'):
        return CATEGORY_TYPE

def walker(graph, start_node, walk_length=100):
    node_now = start_node
    sequence = [node_now]
    # walk_path = get_walk_path(path, walk_length)

    for i in range(walk_length):
        neighbors = [node for node in get_neighbors(graph[node_now]) if get_type(node) < 2]
        if len(neighbors) > 0:
            select = random.choice(neighbors)
            sequence.append(select)
        else:
            return []
        node_now = select

    return sequence

def deepwalk_walker(graph, nodes, walk_times=10, walk_length=50):
    sentences = []
    len_nodes = len(nodes)
    for i, node in enumerate(nodes):
        for _ in range(walk_times):
            sentence = walker(graph, node, walk_length=walk_length)
            if len(sentence) > 0:
                sentences.append(sentence)
        print("{}/{}".format(i, len_nodes), end='\r')
    
    return sentences

if __name__ == '__main__':
    start_time = time()
    PARSER = argparse.ArgumentParser(description='Transform text data to edge list file.')

    PARSER.add_argument('-i', '--input', default=None, help='input file')
    PARSER.add_argument('-o', '--output', default=None, help='output file')

    CONFIG = PARSER.parse_args()
    if CONFIG.input == None:
        print("Please give input file name!")
        exit()
    elif CONFIG.output == None:
        print("Please give output file name!")
        exit()
    graph_path = CONFIG.input
    result_path = CONFIG.output

    print("Reading graph file...")
    graph = nx.read_gpickle(graph_path)

    nodes = graph.nodes()
    user_nodes = [node for node in nodes if node.startswith('u_')]
    item_nodes = [node for node in nodes if node.startswith('i_')]
    word_nodes = [node for node in nodes if node.startswith('w_')]
    category_nodes = [node for node in nodes if node.startswith('c_')]
    print("{} users, {} items, {} words, {} category".format(len(user_nodes), len(item_nodes), len(word_nodes), len(category_nodes)))

    # user_sentences = []
    # item_sentences = []
    # print("Walking user sentences...")
    # for path in user_paths:
        # print('Run the path {}'.format(path))
        # user_sentences.extend(metapath(graph, user_nodes, path, walk_times=WALK_TIMES, walk_length=WALK_LENGTH))
    # print("Walking item sentences...")
    # for path in item_paths:
        # print('Run the path {}'.format(path))
        # item_sentences.extend(metapath(graph, item_nodes, path, walk_times=WALK_TIMES, walk_length=WALK_LENGTH))

    corpus = []
    print("Walking user sentences...")
    corpus.extend(deepwalk_walker(graph, user_nodes, walk_times=WALK_TIMES, walk_length=WALK_LENGTH))
    print("Walking item sentences...")
    corpus.extend(deepwalk_walker(graph, item_nodes, walk_times=WALK_TIMES, walk_length=WALK_LENGTH))
    # print("Walking word sentences...")
    # corpus.extend(deepwalk_walker(graph, word_nodes, walk_times=WALK_TIMES, walk_length=WALK_LENGTH))

    print("Training word2vec...")
    model = Word2Vec(corpus, size=100, window=5, min_count=5, workers=12)
    model.wv.save_word2vec_format(result_path)

    end_time = time()
    print("spend {0:.2f} secs".format(end_time - start_time))
