import networkx as nx
import pickle
import random
from time import time
import argparse

# constant
USER_TYPE = 0
ITEM_TYPE = 1
WORD_TYPE = 2
CATEGORY_TYPE = 3
WALK_TIMES = 50
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

def get_walk_path(path, walk_length):
    sequence = []
    path = [get_type(p) for p in path]
    sequence += path
    
    while len(sequence) < walk_length:
        sequence += path[1:]

    return sequence

def walker(graph, start_node, walk_path, walk_length=100):
    node_now = start_node
    sequence = [node_now]
    # walk_path = get_walk_path(path, walk_length)

    for i in range(len(walk_path) - 1):
        neighbors = [node for node in get_neighbors(graph[node_now]) if get_type(node) == walk_path[i+1]]
        # index = randint(len(neighbors))
        # select = neighbors[index]
        if len(neighbors) > 0:
            select = random.choice(neighbors)
            sequence.append(select)
        else:
            return []
        node_now = select

    return sequence    

def metapath(graph, nodes, path, walk_times=10, walk_length=100):
    sentences = []
    walk_path = get_walk_path(path, walk_length)
    len_nodes = len(nodes)
    for i, node in enumerate(nodes):
        for _ in range(walk_times):
            sentence = ' '.join(walker(graph, node, walk_path, walk_length=walk_length))
            if len(sentence) > 0:
                sentences.append(sentence)
        print("{}/{}".format(i, len_nodes), end='\r')

    return sentences

if __name__ == '__main__':
    start_time = time()
    PARSER = argparse.ArgumentParser(description='Transform text data to edge list file.')

    PARSER.add_argument('-i', '--input', default=None, help='input file')
    PARSER.add_argument('-o', '--output', default=None, help='output file')
    PARSER.add_argument('-w', "--word", type=int, default=0, help='enable word path or not. default is 0.')

    CONFIG = PARSER.parse_args()
    if CONFIG.input == None:
        print("Please give input file name!")
        exit()
    elif CONFIG.output == None:
        print("Please give output file name!")
        exit()
    graph_path = CONFIG.input
    result_path = CONFIG.output
    enable_word = CONFIG.word

    # paths = ['uiu', 'iui', 'uiciu']
    if enable_word:
        # user_paths = ['uiwiu', 'uiu']
        user_paths = ['uiwiu']
    else:
        user_paths = ['uiu']
    item_paths = ['iui']
    # item_paths = ['iui', 'iwi']
    # word_paths = ['wiuiw']

    print("Reading graph file...")
    graph = nx.read_gpickle(graph_path)
    result_path = result_path

    nodes = graph.nodes()
    user_nodes = [node for node in nodes if node.startswith('u_')]
    item_nodes = [node for node in nodes if node.startswith('i_')]
    word_nodes = [node for node in nodes if node.startswith('w_')]
    category_nodes = [node for node in nodes if node.startswith('c_')]
    print("{} users, {} items, {} words, {} category".format(len(user_nodes), len(item_nodes), len(word_nodes), len(category_nodes)))

    user_sentences = []
    item_sentences = []
    word_sentences = []
    print("Walking user sentences...")
    for path in user_paths:
        print('Run the path {}'.format(path))
        user_sentences.extend(metapath(graph, user_nodes, path, walk_times=WALK_TIMES, walk_length=WALK_LENGTH))
    print("Walking item sentences...")
    for path in item_paths:
        print('Run the path {}'.format(path))
        item_sentences.extend(metapath(graph, item_nodes, path, walk_times=WALK_TIMES, walk_length=WALK_LENGTH))
    # print("Walking word sentences...")
    # for path in word_paths:
        # print('Run the path {}'.format(path))
        # word_sentences.extend(metapath(graph, word_nodes, path, walk_times=WALK_TIMES, walk_length=WALK_LENGTH))
    
    with open(result_path, 'w') as f:
        for sentence in user_sentences:
            print(sentence, file=f)
        for sentence in item_sentences:
            print(sentence, file=f)

    end_time = time()
    print("spend {0:.2f} secs".format(end_time - start_time))
