from params import Params as param
import modules as my
import tensorflow as tf


class Graph(object):
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            B, N, M, C = param.batch_size, param.max_context_words, param.max_query_words, param.max_chars
            self.x_w = tf.placeholder(tf.int32, shape=[B, N], name="context_words")
            self.x_c = tf.placeholder(tf.int32, shape=[B, N, C], name="context_word_chars")
            self.q =  tf.placeholder(tf.int32, shape=[B, M], name="question")
            #TODO: check output format
            self.y = tf.placeholder(tf.int32, shape=[B, 2, N], name="out")
             
                       
                       
            #TODO: can explore tf.AUTO_REUSE for weight sharing
            #part1: an embedding layer
            VW, VC, DW, DC = param.word_vocab_size, param.char_vocab_size, param.word_emb_dim, param.char_emb_dim     
            #TODO: is reuse = false valid
            self.x_w_emb = my.embedding(inputs=self.x_w, shape=[VW, DW], scope="word_embedding", reuse=None)
            self.q_emb = my.embedding(inputs=self.q, scope="word_embedding", reuse=True)
            self.x_c_emb = my.embedding(inputs=self.x_c, shape=[VC, DC], scope="char_embedding", reuse=None)
            
            self.x_c_emb = tf.reduce_max(self.x_c_emb, reduction_indices=[2])
            self.x_w_emb = tf.concat(values=[self.x_w_emb, self.x_c_emb], axis=2, name="x_w_emb")
            
            self.x_w_emb = my.highway_network(inputs=self.x_w_emb, layers=param.highway_num_layers, scope='highway_net', reuse=None)
            self.x_q_emb = my.highway_network(inputs=self.x_q_emb, layers=param.highway_num_layers, scope='highway_net',  reuse=True)            
            
            
            
            #part2: an embedding encoder layer
            self.x_w_enc = my.encoder_block(inputs=self.x_w_emb, scope='emb_enc_context', reuse=None, flag_first=True)
            self.q_enc = my.encoder_block(inputs=self.x_w_emb, scope='emb_enc_question', reuse=None, flag_first=True)
            
            
            
            #part3: a context-query attention layer
            self.att_a, self.att_b = my.context_query_attention(input1=self.x_w_enc, input2=self.c_q_enc, scope='context_query_att', reuse=None)
            
            
            
            #part4: a model encoder layer
            self.c_mult_att_a = tf.multiply(self.x_w_enc, self.att_a)
            self.c_mult_att_b = tf.multiply(self.x_w_enc, self.att_b)
            self.model_enc = tf.concat(values=[x_w_enc, att_a, c_mult_att_a, c_mult_att_b], axis=2, name="model_enc_inp")
            for i in range(3):                
                for j in range(7):
                    #the call to the first model encoder block in each stack will have reuse None
                    if (j == 0):
                        self.model_enc = my.encoder_block(inputs=self.model_enc, scope='model_enc_stack_{}'.format(i), reuse=None, flag_first=True)
                    #subsequent blocks in each stack (block 2 to 7) will have reuse True since each stack shares weights across blocks
                    else:
                        self.model_enc = my.encoder_block(inputs=self.model_enc, scope='model_enc_stack_{}'.format(i), reuse=True, flag_first=True)
                if (i == 1):
                    #store model_enc as output M0 after completion of run of first stack of model encoder blocks
                    #model encoder blocks executed: 7
                    #using tf.identity to copy a tensor
                    out_m0 = tf.identity(model_enc)
                    #store model_enc as output M1 after completion of run of second stack of model encoder blocks
                    #model encoder blocks executed: 14
                elif(i==2):
                    out_m1 = tf.identity(model_enc)
                    #store model_enc as output M2 after completion of run of third stack of model encoder blocks
                    #model encoder blocks executed: 21
                else:
                    out_m2 = tf.identity(model_enc)            
            
             
             
            #part5: an output layer
            
            
            
            
            '''
            
            with tf.variable_scope("embedding_layer"):

                self.word_emb_mat = tf.get_variable(name="word_emb_mat", dtype=tf.float32, initializer=tf.random_uniform([VW, DW]), trainable=True)
                self.w_vecs = tf.nn.embedding_lookup(self.word_emb_mat, self.x_w)
                self.q_vecs = tf.nn.embedding_lookup(self.word_emb_mat, self.q)
                
                self.char_emb_mat = tf.get_variable(name="char_emb_mat", dtype=tf.float32, initializer=tf.random_uniform([VC, DC]), trainable=True)
                self.char_vecs = tf.nn.embedding_lookup(self.char_emb_mat, self.x_c)
            '''
                
                
            

                
                
                
            
            
            
            
