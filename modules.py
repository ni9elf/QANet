'''
File modules.py contain helper functions and custom neural layers, written in a modular style. Coupled with the OOP paradigm used to define the computation graph in qanet.py, these functions help in abstracting the complexity of the architecture and Tensorflow features (such as sharing of Tensors) from the computation graph definition in qanet.py. The computation graph can be defined in a simple manner in qanet.py by using calls to functions defined in this file, which take care of the actual complexity.
'''

import tensorflow as tf
import numpy as np


def embedding(inputs, shape=None, scope="word_embedding", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        #TODO: initialize word to glove
        #self.word_emb_mat = tf.get_variable(name="word_emb_mat", dtype=tf.float32, shape=[VW, DW], initializer=tf.constant_initializer(self.word_vocab_matrix), trainable=True)   
        #self.word_vocab_matrix = load_word_vocab()  
        emb_matrix = tf.get_variable('word_emb_matrix', dtype=tf.float32, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
        outputs = tf.nn.embedding_lookup(emb_matrix, inputs)
        return outputs


def highway_layer(inputs, use_bias, transform_bias=-1.0, scope="highway_layer", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        dims = inputs.get_shape()[-1]
        #TODO: proper dense W and B initializers using glorot?
        z = tf.layers.dense(inputs, dims, use_bias=use_bias, name='highway_dense_1', reuse=reuse)
        activation = tf.nn.relu(z)
        transform_gate = tf.layers.dense(inputs, dims, use_bias=use_bias, bias_initializer=tf.constant_initializer(transform_bias), name='highway_dense_2', reuse=reuse)
        transform_gate = tf.nn.sigmoid(transform_gate)
        outputs = transform_gate * activation + (1 - transform_gate) * inputs
        return outputs                    


def highway_network(inputs, num_layers=2, use_bias=True, transform_bias=-1.0, scope='highway_net', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        for layer_id in range(num_layers):
            inputs = highway_layer(inputs, use_bias, transform_bias, scope='highway_layer_{}'.format(layer_id), reuse=reuse)
        return inputs         


def positional_encoding(inputs, scope='positional_enc', reuse=None):
    batch_size, max_length, dims = inputs.get_shape().as_list()   
    with tf.variable_scope(scope, reuse=reuse):
        #create index matrix of shape=[batch_size, max_length] for future lookup
        pos_ind = tf.tile(tf.expand_dims(tf.range(max_length), 0), [batch_size, 1])
        
        #creating position encoding matrix
        pos_enc_matrix = np.array([[pos / np.power(10000, 2.*i/dims) for i in range(dims)] for pos in range(max_length)])
        #applying sin to odd columns
        pos_enc_matrix[:, 0::2] = np.sin(pos_enc_matrix[:, 0::2]) 
        #applying cosine to even columns
        pos_enc_matrix[:, 1::2] = np.cos(pos_enc_matrix[:, 1::2])
        pos_enc_matrix = tf.convert_to_tensor(pos_enc_matrix, dtype=tf.float32)
        
        outputs = tf.nn.embedding_lookup(pos_enc_matrix, pos_ind)
        return outputs
                

def convolution(inputs, filters, kernel_size, scope, reuse=None):
    #TODO: to reuse layers is it ok to put reuse outside in variable scope, or should reuse be put inside the layer call with same name, or use layers which have scope also    
    with tf.variable_scope(scope, reuse=reuse): 
        #layernorm
        outputs = tf.contrib.layers.layer_norm(inputs, scope='norm_'+scope, reuse=reuse)
        #1D convolution
        outputs = tf.layers.conv1d(outputs, filters, kernel_size, padding="same", name='conv_'+scope, reuse=reuse)        
        #if inputs are compatible with outputs for residual link
        if(inputs.get_shape()[-1] == outputs.get_shape()[-1]):
            #residual link
            outputs += inputs
        return outputs
                                                               

def multi_head_attention(queries, keys, values, num_heads=8, scope='multi_head_attention', reuse=None):
    #applies a multi head attention in a self attention fashion
    with tf.variable_scope(scope, reuse=reuse): 
        Q = queries
        K = keys
        V = values
        dims = queries.get_shape().as_list()[-1] / num_heads
        #split into # of head parts
        Q_s = tf.split(Q, num_heads, axis=2)
        K_s = tf.split(K, num_heads, axis=2)
        V_s = tf.split(V, num_heads, axis=2)
        #project using different learned linear projections
        Q_s = [tf.layers.dense(q, dims, activation=tf.nn.relu) for q in Q_s]
        K_s = [tf.layers.dense(q, dims, activation=tf.nn.relu) for k in K_s]
        V_s = [tf.layers.dense(q, dims, activation=tf.nn.relu) for v in V_s]
        #concatenate different projections for parallel scaled dot product attention
        Q_c = tf.concat(Q_s, axis=0)
        K_c = tf.concat(K_s, axis=0)
        V_c = tf.concat(V_s, axis=0)
        #perform a scaled dot product attention in parallel for all heads
        outputs = tf.matmul(Q_c, tf.transpose(K_c, [0, 2, 1]))
        #scale outputs using square_root(K.shape[-1])
        outputs = outputs / (K_s[0].get_shape().as_list()[-1] ** 0.5)
        #applying softmax normalization
        outputs = tf.nn.softmax(outputs)
        #applying weights on values
        #restore shape of values to original input shape
        outputs = tf.matmul(outputs, V_c)
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 )        
        #residual link
        outputs += queries
        return outputs


def feedforward(inputs, scope='feedforward', reuse=None):
    with tf.variable_scope(scope, reuse=reuse): 
        #layernorm
        outputs = tf.contrib.layers.layer_norm(inputs, scope='norm_'+scope, reuse=reuse)
        #dense layer
        dims = outputs.get_shape()[-1]        
        outputs = tf.layers.dense(outputs, dims, activation=tf.nn.relu, name='dense_'+scope, reuse=reuse)
        #residual link
        outputs += inputs
        return outputs


def encoder_block(inputs, num_conv_layer=4, filters=128, kernel_size=7, num_att_head=8, scope='encoder_block', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        #add positional encoding
        inputs += positional_encoding(inputs, scope='positional_enc', reuse=reuse)
        #convolution layers        
        for layer_id in range(num_conv_layer):
            inputs = convolution(inputs, filters, kernel_size, scope='layer_{}'.format(layer_id), reuse=reuse)
        #self-attention using multi head attention layer                
        outputs = multi_head_attention(queries=inputs, keys=inputs, values=inputs, num_heads=num_att_head, scope='multi_head_attention', reuse=reuse)
        #feedforward layer
        outputs = feedforward(outputs, scope='feedforward', reuse=reuse)
        return outputs
        

def context_query_attention(context, query, scope='context_query_att', reuse=None):
    #batch_size, max_words_context, word_dimension
    B, N, d = context.get_shape().as_list()
    #batch_size, max_words_question, word_dimension
    B, M, d = query.get_shape().as_list()
    with tf.variable_scope(scope, reuse=reuse): 
        #[B, N, d] -> [B, N, M, d]
        context_expand = tf.tile(tf.expand_dims(context, 2), [1, 1, M, 1])
        #[B, M, d] -> [B, N, M, d]
        query_expand = tf.tile(tf.expand_dims(query, 1), [1, N, 1, 1])
        #concat(q, c, (q)dot(c))
        mat = tf.concat((query_expand, context_expand, query_expand * context_expand), axis=3)
        #trilinear function as a linear dense layer
        #TODO: no need to give 'dense_'+scope, just name since scope/name automatic
        similarity = tf.layers.dense(mat, 1, name='dense_'+scope, reuse=reuse)
        similarity = tf.squeeze(similarity)
        matrix_a = tf.matmul(similarity, query)
        matrix_b = tf.matmul(tf.matmul(similarity, tf.transpose(similarity, [0, 2, 1])), context)
        return matrix_a, matrix_b
