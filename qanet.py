'''
File qanet.py defines the Tensorflow computation graph for the QANet architecture using OOP and modularity. The skeleton of the architecture from inputs to outputs in defined here using calls to wrapper functions defined in modules.py to take care of the inner working of each component. This makes the graph definition easy to code, read and understand. The advantages of OOP, especially abstraction of weight / Tensor sharing and encapsulation of sub component / layer inner working can be realized. Modularity ensures that the functioning of a component can be easily modified in modules.py without changing the skeleton of the QANet architecture defined in this file.

Author: Nigel Fernandez
'''

from params import Params as param
import modules as my
import tensorflow as tf
import numpy as np


#a tensorflow computation graph is treated as an object of the Graph class
class Graph(object):
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            #paceholders for inputs and outputs
            B, N, M, C = param.batch_size, param.max_context_words, param.max_question_words, param.max_chars
            #inputs
            #input sequence of word vocabulary indices of the context
            self.x_c_w = tf.placeholder(tf.int32, shape=[B, N], name="context_words")
            #input sequence of char vocabulary indices (0 to 25) of the words of the context
            self.x_c_c = tf.placeholder(tf.int32, shape=[B, N, C], name="context_word_chars")
            #input sequence of question vocabulary indices of the context
            self.x_q_w =  tf.placeholder(tf.int32, shape=[B, M], name="question_words")
            #input sequence of char vocabulary indices (0 to 25) of the words of the question
            self.x_q_c = tf.placeholder(tf.int32, shape=[B, M, C], name="context_question_chars")
            #output as a one hot encoding of the start position and end position indices over the context words
            self.y = tf.placeholder(tf.int32, shape=[B, N, 2], name="out")
             
                       
            '''          
            part1: an embedding layer
            '''
            VW, VC, DW, DC = param.word_vocab_size, param.char_vocab_size, param.word_emb_dim, param.char_emb_dim     
            self.x_c_w_emb = my.embedding(inputs=self.x_c_w, shape=[VW, DW], scope="word_embedding", reuse=None)
            self.x_q_w_emb = my.embedding(inputs=self.x_q_w, scope="word_embedding", reuse=True)
            self.x_c_c_emb = my.embedding(inputs=self.x_c_c, shape=[VC, DC], scope="char_embedding", reuse=None)
            self.x_q_c_emb = my.embedding(inputs=self.x_q_c, scope="char_embedding", reuse=True)
            
            #max pooling
            self.x_c_c_emb = tf.reduce_max(self.x_c_c_emb, reduction_indices=[2])
            self.x_c_emb = tf.concat(values=[self.x_c_w_emb, self.x_c_c_emb], axis=2, name="x_context_emb")
            self.x_q_c_emb = tf.reduce_max(self.x_q_c_emb, reduction_indices=[2])
            self.x_q_emb = tf.concat(values=[self.x_q_w_emb, self.x_q_c_emb], axis=2, name="x_question_emb")            
            
            #highway network of 2 layers
            self.x_c_emb = my.highway_network(inputs=self.x_c_emb, num_layers=param.highway_num_layers, use_bias=True, transform_bias=-1.0, scope='highway_net', reuse=None)
            self.x_q_emb = my.highway_network(inputs=self.x_q_emb, num_layers=param.highway_num_layers, use_bias=True, transform_bias=-1.0, scope='highway_net',  reuse=True)            
            
            
            '''
            part2: an embedding encoder layer
            '''
            self.x_c_enc = my.encoder_block(inputs=self.x_c_emb, num_conv_layer=4, filters=128, kernel_size=7, num_att_head=8, scope='encoder_block', reuse=None)
            self.x_q_enc = my.encoder_block(inputs=self.x_q_emb, num_conv_layer=4, filters=128, kernel_size=7, num_att_head=8, scope='encoder_block', reuse=True)
            
            
            '''           
            part3: a context-query attention layer
            '''
            self.att_a, self.att_b = my.context_query_attention(context=self.x_c_enc, query=self.x_q_enc, scope='context_query_att', reuse=None)
            
            
            '''
            part4: a model encoder layer
            ''' 
            self.c_mult_att_a = tf.multiply(self.x_c_enc, self.att_a)
            self.c_mult_att_b = tf.multiply(self.x_c_enc, self.att_b)
            self.model_enc = tf.reduce_mean(tf.concat([tf.expand_dims(self.x_c_enc, 2), tf.expand_dims(self.att_a, 2), tf.expand_dims(self.c_mult_att_a, 2), tf.expand_dims(self.c_mult_att_b, 2)], axis=2), axis=2, name="model_enc_inp")            
            for i in range(3):                
                for j in range(7):
                    #the call to the first model encoder block in each stack will have reuse None
                    if (i == 0):
                        self.model_enc = my.encoder_block(inputs=self.model_enc, num_conv_layer=2, filters=128, kernel_size=5, num_att_head=8, scope='model_enc_block_{}'.format(j), reuse=None)
                    #subsequent blocks in each stack (block 2 to 7) will have reuse True since each stack shares weights across blocks
                    else:
                        self.model_enc = my.encoder_block(inputs=self.model_enc, num_conv_layer=2, filters=128, kernel_size=5, num_att_head=8, scope='model_enc_block_{}'.format(j), reuse=True)
                if (i == 1):
                    #store model_enc as output M0 after completion of run of first stack of model encoder blocks
                    #model encoder blocks executed: 7
                    #using tf.identity to copy a tensor
                    self.out_m0 = tf.identity(self.model_enc)
                    #store model_enc as output M1 after completion of run of second stack of model encoder blocks
                    #model encoder blocks executed: 14
                elif(i==2):
                    self.out_m1 = tf.identity(self.model_enc)
                    #store model_enc as output M2 after completion of run of third stack of model encoder blocks
                    #model encoder blocks executed: 21
                else:
                    self.out_m2 = tf.identity(self.model_enc)            
                    
                    
            '''        
            part5: an output layer      
            '''
            self.inp_pos1 = tf.concat((self.out_m0, self.out_m1), axis=2)
            self.inp_pos2 = tf.concat((self.out_m0, self.out_m2), axis=2)                             
            self.pos1 = tf.nn.softmax(tf.layers.dense(self.inp_pos1, 1, activation=tf.tanh, name='dense_pos1'))
            self.pos2 = tf.nn.softmax(tf.layers.dense(self.inp_pos2, 1, activation=tf.tanh, name='dense_pos2'))
            self.pred = tf.concat((self.pos1, self.pos2), axis = -1)
            self.loss = tf.reduce_mean(-tf.log(tf.reduce_prod(tf.reduce_sum(self.pred * tf.cast(self.y, 'float'), 1), 1) + param.epsilon_1))            
            
            #training scheme
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=param.lr, beta1=param.beta1, beta2=param.beta2, epsilon=param.epsilon_2)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
            tf.summary.scalar('loss', self.loss)
            self.merged = tf.summary.merge_all()
