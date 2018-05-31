import tensorflow as tf

def load_word_vocab():
    pass


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
        z = tf.layers.dense(inputs, dims, use_bias=use_bias)
        activation = tf.nn.relu(z)
        transform_gate = tf.layers.dense(inputs, dims, use_bias=use_bias, bias_initializer=tf.constant_initializer(transform_bias))
        transform_gate = tf.nn.sigmoid(transform_gate)
        outputs = transform_gate * activation + (1 - transform_gate) * activation
        return outputs                    


def highway_network(inputs, num_layers=2, use_bias=True, transform_bias=-1.0, scope='highway_net', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        for layer_id in range(num_layers):
            inputs = highway_layer(inputs, use_bias, transform_bias, scope='highway_layer_{}'.format(layer_id), reuse=reuse)
        return inputs            
