#!/bin/bash -x

##############################

# 1. Preprocessing 

##############################

# split train/test into json file
python3 ./preprocessing/train_test_amazon.py ./data/reviews_Cell_Phones.json.gz ./data/train.json ./data/test.json
# construct network
python3 ./preprocessing/graph_amazon.py ./data/train.json ./data/meta_Cell_Phones.json.gz ./data/network_Cell_Phones.pickle

##############################

# 2. Network Embedding

##############################

cd network_embedding/code_metapath2vec
make
cd ../proNet-core
make
cd ../..
# construct edgelist
python3 ./network_embedding/edgelist.py --input ./data/network_Cell_Phones.pickle --output ./data/edgelist_Cell_Phones.txt --word 1
# train Deepwalk
./network_embedding/proNet-core/cli/deepwalk -train ./data/edgelist_Cell_Phones.txt -save ./data/Cell_Phones_100_deepwalk_pronet.embd.txt -walk_times 10 -walk_steps 50 -dimensions 100 -threads 4
# train LINE
./network_embedding/proNet-core/cli/line -train ./data/edgelist_Cell_Phones.txt -save ./data/Cell_Phones_100_line.embd.txt -sample_times 200 -dimensions 100 -threads 4
# train HPE
./network_embedding/proNet-core/cli/hpe -train ./data/edgelist_Cell_Phones.txt -save ./data/Cell_Phones_100_hpe.embd.txt -sample_times 200 -dimensions 100 -threads 4

# construct metapath corpus
python3 metapath_walker.py --input ./data/network_Cell_Phones.pickle --output ./data/metapath_Cell_Phones.txt --word 1
./code_metapath2vec/metapath2vec -train ./data/metapath_Cell_Phones.txt -output ./data/Cell_Phones_100_metapath.embd.txt

##############################

# 3. Recommendation Model

##############################

python3 ./rec_model/tf_rec.py ./data/train.json ./data/test.json ./data/Cell_Phones_100_deepwalk_pronet.embd.txt ./data/pred.pickle

