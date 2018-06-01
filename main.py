from qanet import Graph
import tensorflow as tf

if __name__ == '__main__': 
    g = Graph("train")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())         
