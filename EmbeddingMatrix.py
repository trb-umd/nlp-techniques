import numpy as np

def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1
    # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    f = open(filepath, encoding="utf8")
    for line in f:
        word, *vector = line.split()
        if word in word_index:
            #print("TRUE")
            idx = word_index[word]
            embedding_matrix[idx] = np.array(vector, dtype=np.float32)[
                                    :embedding_dim]
    return embedding_matrix