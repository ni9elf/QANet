'''
File params.py defines a class Params for hyperparameter declarations.

Author: Nigel Fernandez
'''

class Params(object):
    batch_size = 32
    max_context_words = 400
    max_query_words = 30
    word_vocab_size = 10000
    char_vocab_size = 26
    word_emb_dim = 300
    char_emb_dim = 200
    max_chars = 16
    highway_num_layers = 2
    epsilon_1 = 1e-30
    epsilon_2 = 1e-7
    beta1 = 0.8
    beta2 = 0.999
    lr = 1e-2
