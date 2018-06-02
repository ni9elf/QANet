'''
File main.py contains the main program. The computation graph for QANet is built here and launched in a session for training.

Author: Nigel Fernandez
'''

from qanet import Graph
import tensorflow as tf

if __name__ == '__main__': 
    #build the computation graph
    g = Graph("train")
    #launch the computation graph in a session
    with tf.Session() as sess:
        #initialize all global variables in the graph
        sess.run(tf.global_variables_initializer())         
