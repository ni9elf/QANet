'''
File main.py contains the main program. The computation graph for QANet is built here and launched in a session.

Author: Nigel Fernandez
'''

from qanet import Graph
import tensorflow as tf
from params import Params as param
from tqdm import tqdm

if __name__ == '__main__': 
    #build the computation graph
    g = Graph()
    print "\n###Computation graph for QANet loaded###\n"
    
    print "\n###Training started###\n"
    #creating training helper
    sv = tf.train.Supervisor(graph=g.graph, logdir=param.logdir)        
    #launch the computation graph in a session
    num_batch = param.num_samples // param.batch_size
    with sv.managed_session() as sess:
        #training for num_epochs
        for epoch in range(1, param.num_epochs+1): 
            #if exception arises
            if sv.should_stop(): 
                break           
            #training progress in current epoch         
            for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
                sess.run(g.train_op)            
            gs = sess.run(g.global_step)
            #saving
            sv.saver.save(sess, param.logdir + '/model_epoch_%02d_gs_%d' % (epoch, gs))
    print "\n###Training complete###\n"                                
