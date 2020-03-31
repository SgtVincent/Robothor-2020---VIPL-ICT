import os
import sys
import json
import numpy as np
sys.path.append('.')
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from dataset import Dictionary
# dbg = IPython.core.debugger.Pdb()



def create_glove_embedding_init(idx2word, glove_file):
    '''
    :param idx2word: a dictionary or list to store the vocabulary, e.g. {0: 'apple', 1: 'cake'} or ['apple', 'cake', ...]
    :param glove_file: the path of glove file
    :return: weights: word embedding matrix
    '''
    word2emb = {}
    with open(glove_file, 'r') as f:
        entries = f.readlines()
    emb_dim = len(entries[0].split(' ')) - 1
    print('embedding dim is %d' % emb_dim)
    weights = np.zeros((len(idx2word), emb_dim), dtype=np.float32)

    for entry in entries:
        vals = entry.split(' ')
        word = vals[0]
        vals = map(float, vals[1:])
        word2emb[word] = np.array(list(vals))   # use list(vals) in python 3

    if type(idx2word) is list:
        idx2word = {i: w for i, w in enumerate(idx2word)}

    for idx, word in idx2word.items():
        if word not in word2emb:
            continue
        else:
            weights[idx] = word2emb[word]
    return weights

if __name__ == '__main__':
    create_glove_embedding_init()