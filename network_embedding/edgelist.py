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

def construct_edgelist(graph, nodes, use_word=True):
    if use_word:
        type_num = 3
    else:
        type_num = 2

    edgelist = []
    for i, node in enumerate(nodes):
        neighbors = [node for node in get_neighbors(graph[node]) if get_type(node) < type_num]
        if len(neighbors) > 0:
            for n in neighbors:
                edgelist.append(((node, n)))
    return edgelist

if __name__ == '__main__':
    start_time = time()
    PARSER = argparse.ArgumentParser(description='Transform text data to edge list file.')

    PARSER.add_argument('-i', '--input', default=None, help='input file')
    PARSER.add_argument('-o', '--output', default=None, help='output file')
    PARSER.add_argument('-w', '--word', default='1', help='use word')

    CONFIG = PARSER.parse_args()
    if CONFIG.input == None:
        print("Please give input file name!")
        exit()
    elif CONFIG.output == None:
        print("Please give output file name!")
        exit()
    graph_path = CONFIG.input
    result_path = CONFIG.output
    if CONFIG.word == '1':
        use_word = True
    else:
        use_word = False

    print("Reading graph file...")
    graph = nx.read_gpickle(graph_path)

    nodes = graph.nodes()
    user_nodes = [node for node in nodes if node.startswith('u_')]
    item_nodes = [node for node in nodes if node.startswith('i_')]
    word_nodes = [node for node in nodes if node.startswith('w_')]
    category_nodes = [node for node in nodes if node.startswith('c_')]
    print("{} users, {} items, {} words, {} category".format(len(user_nodes), len(item_nodes), len(word_nodes), len(category_nodes)))

    edgelist = []
    print("Append user edgelist...")
    edgelist.extend(construct_edgelist(graph, user_nodes, use_word=use_word))
    print("Appedn item edgelist...")
    edgelist.extend(construct_edgelist(graph, item_nodes, use_word=use_word))
    # print("Walking word sentences...")
    # corpus.extend(deepwalk_walker(graph, word_nodes, walk_times=WALK_TIMES, walk_length=WALK_LENGTH))

    with open(result_path, 'w') as f:
        for edge in edgelist:
            f.write(edge[0] + ' ' + edge[1] + ' 1\n')

    end_time = time()
    print("spend {0:.2f} secs".format(end_time - start_time))
