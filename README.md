# master_thesis

My master thesis: E-commerce Recommendation Systems Based on Heterogeneous Information Network Embedding

## Introduction

This paper incorporates words from product title as the attributes of the item. Then, we transform words, and user behavior into heterogeneous network for E-commerce. For this network, we use various network embedding methods to learn both user and item representations in the same latent space. Moreover, we integrate the learned embedding as the features into Matrix Factorization . 


## Get Amazon Data

Download [review](https://drive.google.com/file/d/1u-zG2k5ZCzpzkRVHF0SpATTBoqc-DpLm/view?usp=sharing) and [metadata](https://drive.google.com/file/d/1u-zG2k5ZCzpzkRVHF0SpATTBoqc-DpLm/view?usp=sharing) example files, and put them in `data` directory. Of course, you can get the 5-core and metadata files with other categories from [McAuley's site](http://jmcauley.ucsd.edu/data/amazon/).

## Dependencies

```
pip install numpy
pip install scipy
pip install nltk
pip install networkx
pip install gensim
pip install lightfm
```

Packages below has been in repo.

[metapath2vec](https://ericdongyx.github.io/metapath2vec/m2v.html), [ProNet-core](https://github.com/cnclabs/proNet-core), [tensorrec](https://github.com/jfkirk/tensorrec)

## Run Script

```
./run.sh
```