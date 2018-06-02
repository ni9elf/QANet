'''
File modules.py contain helper functions and custom neural layers, written in a modular style. Coupled with the OOP paradigm used to define the computation graph in qanet.py, these functions help in abstracting the complexity of the architecture and Tensorflow features (such as sharing of Tensors) from the computation graph definition in qanet.py. The computation graph can be defined in a simple manner in qanet.py by using calls to functions defined in this file, which take care of the actual complexity.
'''

import tensorflow as tf
import numpy as np


def embedding(inputs, shape=None, scope="word_embedding", reuse=None):
    '''
    Defines an embedding layer
    '''
    with tf.variable_scope(scope, reuse=reuse):
        #create and initialize an embedding matrix (lookup table) of shape [vocab_size, emb_dims]
        #if called with reuse=None, a new embedding matrix is created 
        #if called with resus=True the existing embedding matrix present in the scope will be reused
        emb_matrix = tf.get_variable("word_emb_matrix", dtype=tf.float32, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
        #get the embeddings for input sequence of indices
        #[..., N] -> [..., N, d] where N is the index sequence length and d is the embedding dimension
        outputs = tf.nn.embedding_lookup(emb_matrix, inputs)
        return outputs


def highway_layer(inputs, use_bias, transform_bias=-1.0, scope="highway_layer", reuse=None):
    '''
    Defines a single highway layer of a highway network
    '''
    with tf.variable_scope(scope, reuse=reuse):
        #get hidden dimension d which is 128 in qanet
        dims = inputs.get_shape()[-1]
        #compute the activation using a dense layer with relu activation
        z = tf.layers.dense(inputs, dims, use_bias=use_bias, name="highway_dense_1", reuse=reuse)
        activation = tf.nn.relu(z)
        #compute the transform gate value using a dense layer with sigmoid activation
        transform_gate = tf.layers.dense(inputs, dims, use_bias=use_bias, bias_initializer=tf.constant_initializer(transform_bias), name='highway_dense_2', reuse=reuse)
        transform_gate = tf.nn.sigmoid(transform_gate)
        #apply the highway network equation: (transform_gate * activation) + (carry_gate * inputs) 
        #carry_gate = (1 - transform_gate)
        outputs = transform_gate * activation + (1 - transform_gate) * inputs
        return outputs                    


def highway_network(inputs, num_layers=2, use_bias=True, transform_bias=-1.0, scope="highway_net", reuse=None):
    '''
    Defines a highway network of num_layers and calls highway_layer to construct each layer
    '''
    with tf.variable_scope(scope, reuse=reuse):
        for layer_id in range(num_layers):
            #call highway_layer in scope "highway_layer_i", if called again using the same scope, layer i will get reused in scope "highway_layer_i"
            #the highway network is reused on both context and question embedding
            inputs = highway_layer(inputs, use_bias, transform_bias, scope="highway_layer_{}".format(layer_id), reuse=reuse)
        return inputs         


def positional_encoding(inputs, scope="positional_enc", reuse=None):
    '''
    Defines a positional encoding layer
    '''
    #get batch_size, max_length, dims from shape of inputs
    batch_size, max_length, dims = inputs.get_shape().as_list()   
    with tf.variable_scope(scope, reuse=reuse):
        #create a position index matrix of shape=[batch_size, max_length] for future lookup from position embedding matrix
        #tf.range(max_length) populates a single row of column size max_length, while tf.tile replicates the single row batch_size times
        pos_ind = tf.tile(tf.expand_dims(tf.range(max_length), 0), [batch_size, 1])        
        #create a position embedding matrix 
        #define various frequencies computes using position of the word and index of the dimension
        pos_enc_matrix = np.array([[pos / np.power(10000, 2.*i/dims) for i in range(dims)] for pos in range(max_length)])        
        #applying sin wave to odd indices of the hidden state
        pos_enc_matrix[:, 0::2] = np.sin(pos_enc_matrix[:, 0::2]) 
        #applying cosine wave to even indices of the hidden state
        pos_enc_matrix[:, 1::2] = np.cos(pos_enc_matrix[:, 1::2])
        #convert position embedding matrix  to a tf.float32 tensor
        pos_enc_matrix = tf.convert_to_tensor(pos_enc_matrix, dtype=tf.float32)
        #get the position embeddings for input sequence of position indices
        #[..., N] -> [..., N, d] where N is the index sequence length and d is the embedding dimension
        outputs = tf.nn.embedding_lookup(pos_enc_matrix, pos_ind)
        return outputs
                

def convolution(inputs, filters, kernel_size, scope, reuse=None):   
    '''
    Defines a convolution layer with inputs first passed to a layernorm
    '''
    with tf.variable_scope(scope, reuse=reuse): 
        #use layernorm before applying convolution
        outputs = tf.contrib.layers.layer_norm(inputs, scope="layernorm", reuse=reuse)
        #perform a 1D convolution
        outputs = tf.layers.conv1d(outputs, filters, kernel_size, padding="same", name="convolution", reuse=reuse)                
        #ATTENTION: there is an ambiguity in the paper here, the input dimension of the hidden state of each wod to the first convolution layer of either the context of question encoder block is 500, while the output after convolution will map the hidden state to 128 dimenstion, therefore a residual link cannot be computed due to a dimension mismatch (500 != 128)
        #if inputs are compatible with outputs then create a residual link
        if(inputs.get_shape()[-1] == outputs.get_shape()[-1]):
            #residual link
            outputs += inputs
        return outputs
                                                               

def multi_head_attention(queries, keys, values, num_heads=8, scope="multi_head_attention", reuse=None):
    '''
    Defines a multi head attention layer
    '''
    #applies a multi head attention in a self attention fashion since queries, keys and values in QANet are the same Tensor
    with tf.variable_scope(scope, reuse=reuse): 
        #use layernorm before applying multi_head_attention
        #if all inputs are equal implying self attention, perform layernorm on any one input tensor
        if (tf.equal(queries, keys) and tf.equal(queries, values) and tf.equal(keys, values)):        
            queries = tf.contrib.layers.layer_norm(queries, scope="layernorm", reuse=reuse)
            #dimension=[B, N, d] ([batch_size, max_seq_length, hidden_state_dimension])
            Q = queries
            K = queries
            V = queries        
        else:
            queries = tf.contrib.layers.layer_norm(queries, scope="layernorm", reuse=reuse)
            keys = tf.contrib.layers.layer_norm(keys, scope="layernorm", reuse=reuse)
            values = tf.contrib.layers.layer_norm(values, scope="layernorm", reuse=reuse)
            #dimension=[B, N, d] ([batch_size, max_seq_length, hidden_state_dimension])
            Q = queries
            K = keys
            V = values
        #compute the dimension of each head (parallel attention layer)
        #in QANet this will be (hidden_state_dimension / num_heads) = 128 / 8 = 16
        dims = queries.get_shape().as_list()[-1] / num_heads
        #split each input tensor into num_head parts (into 8 parts)
        #we split each tensor to ensure computation cost remains same even though num_head attention layers are called
        #dimensions=[h, B, N, d/h] ([num_heads, batch_size, max_seq_length, hidden_state_dimension / num_heads])
        #note that tf.split returns a sequence of tensors
        Q_s = tf.split(Q, num_heads, axis=2)
        K_s = tf.split(K, num_heads, axis=2)
        V_s = tf.split(V, num_heads, axis=2)
        #project using different learned linear projections
        #dimensions=[h, B, N, d/h]
        Q_s = [tf.layers.dense(q, dims, activation=tf.nn.relu) for q in Q_s]
        K_s = [tf.layers.dense(q, dims, activation=tf.nn.relu) for k in K_s]
        V_s = [tf.layers.dense(q, dims, activation=tf.nn.relu) for v in V_s]
        #concatenate different projections for parallel scaled dot product attention
        #dimensions=[h*B, N, d/h]
        Q_c = tf.concat(Q_s, axis=0)
        K_c = tf.concat(K_s, axis=0)
        V_c = tf.concat(V_s, axis=0)
        #perform a scaled dot product attention in parallel for all heads
        #dimensions=[h*B, N, N]
        outputs = tf.matmul(Q_c, tf.transpose(K_c, [0, 2, 1]))
        #scale outputs using square_root(K.shape[-1])
        #dimensions=[h*B, N, N]
        outputs = outputs / (K_s[0].get_shape().as_list()[-1] ** 0.5)
        #applying softmax normalization
        #dimensions=[h*B, N, N]
        outputs = tf.nn.softmax(outputs)
        #applying weights on values        
        #dimensions=[h*B, N, d/h]
        outputs = tf.matmul(outputs, V_c)
        #restore shape of values to original input shape
        #dimensions=[B, N, d]
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)        
        #use a residual link
        outputs += queries
        return outputs


def feedforward(inputs, scope="feedforward", reuse=None):
    '''
    Defines a feedforward layer
    '''
    with tf.variable_scope(scope, reuse=reuse): 
        #layernorm
        outputs = tf.contrib.layers.layer_norm(inputs, scope="layernorm", reuse=reuse)
        #dense layer
        dims = outputs.get_shape()[-1]        
        outputs = tf.layers.dense(outputs, dims, activation=tf.nn.relu, name="dense", reuse=reuse)
        #residual link
        outputs += inputs
        return outputs


def encoder_block(inputs, num_conv_layer=4, filters=128, kernel_size=7, num_att_head=8, scope="encoder_block", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        #add positional encoding
        inputs += positional_encoding(inputs, scope="positional_enc", reuse=reuse)
        #convolution layers        
        for layer_id in range(num_conv_layer):
            inputs = convolution(inputs, filters, kernel_size, scope="layer_{}".format(layer_id), reuse=reuse)
        #self-attention using multi head attention layer                
        outputs = multi_head_attention(queries=inputs, keys=inputs, values=inputs, num_heads=num_att_head, scope="multi_head_attention", reuse=reuse)
        #feedforward layer
        outputs = feedforward(outputs, scope="feedforward", reuse=reuse)
        return outputs
        

def context_query_attention(context, query, scope="context_query_att", reuse=None):
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
        similarity = tf.layers.dense(mat, 1, name="dense", reuse=reuse)
        similarity = tf.squeeze(similarity)
        matrix_a = tf.matmul(similarity, query)
        matrix_b = tf.matmul(tf.matmul(similarity, tf.transpose(similarity, [0, 2, 1])), context)
        return matrix_a, matrix_b
