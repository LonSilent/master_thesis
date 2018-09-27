from gensim.models.keyedvectors import KeyedVectors

embd_path = 'out.txt.txt'

word_vectors = KeyedVectors.load_word2vec_format(embd_path, binary=False)
print(word_vectors.similar_by_word('vKDD'))