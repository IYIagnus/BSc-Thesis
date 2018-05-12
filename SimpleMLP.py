import numpy as np
import tensorflow as tf
import pandas as pd
from DataProcessing import *

import sys
import time

class Config(object):
    """hyperparameters and data"""
    batch_size = 20
    max_epochs = 256
    hidden_size = 100
    lr = 0.001
    training_size = 5
    valid_size = 2
    num_runs = 1
    #data = getFormattedData()
    
class MultiLayerPerceptron(object):
    def __init__(self, config):
        self.config = config
        self.W1 = tf.Variable(tf.truncated_normal([len(self.config.data[0][0]) - 1, self.config.hidden_size], stddev=0.1))
        self.W2 = tf.Variable(tf.truncated_normal([self.config.hidden_size, 1], stddev=0.1))
        self.b1 = tf.Variable(tf.zeros(self.config.hidden_size))
        self.b2 = tf.Variable(tf.ones(1))
        self.add_placeholders()
        self.output = self.inference()
        self.loss = self.get_loss()
        self.train_step = self.add_training_op(self.loss)
        self.merged = tf.summary.merge_all()

        
    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.float32, [self.config.batch_size, 5], "input_placeholder")
        self.labels = tf.placeholder(tf.float32, [self.config.batch_size,], "labels")
        
    def inference(self):
        with tf.variable_scope("hidden_layer"):
            hidden_layer = tf.sigmoid(tf.matmul(self.input_placeholder, self.W1) + self.b1)
        with tf.variable_scope("output_layer"):
            output = tf.nn.relu(tf.matmul(hidden_layer, self.W2) + self.b2)

        return output
        
    def get_loss(self):
        with tf.variable_scope("loss_function"):
            loss = tf.losses.mean_squared_error(tf.reshape(self.labels, [self.config.batch_size, 1]), self.output)
        
            tf.summary.scalar('loss', loss)

        
        return loss
    
    def add_training_op(self, loss):
        """Calculate and apply gradients"""
        with tf.variable_scope("optimizer_and_gradients"):
            opt = tf.train.AdamOptimizer(learning_rate=self.config.lr)
            gvs = opt.compute_gradients(loss)
            
            train_op = opt.apply_gradients(gvs)
            
            return train_op
        
    
    def run_epoch(self, session, data, num_epoch=0, train_writer=None, train_op=None):
        config = self.config
        total_steps = len(data) // config.batch_size
        
        if train_op is None:
            train_op = tf.no_op()
            
        total_loss = []
        
        labels = data[:, 5]
        inputs = data[:, [0, 1, 2, 3, 4]]
        
        p = np.random.permutation(len(labels))
        labels, inputs = labels[p], inputs[p]
    
        #print(data)
        
        for step in range(total_steps):
            index = range(step*config.batch_size,(step+1)*config.batch_size)
            feed_dict = {self.input_placeholder: inputs[index].astype(np.float32),
                         self.labels: labels[index].astype(np.float32),
                         }
            loss, summary, _ = session.run([self.loss, self.merged, train_op], feed_dict=feed_dict)
            
            total_loss.append(loss)
            
            if not(train_writer is None):
                train_writer.add_summary(summary, num_epoch*total_steps + step)
            
            if step % 2 == 0:
                sys.stdout.write('\r{} / {} : loss = {}'.format(
                  step, total_steps, np.mean(total_loss)))
                sys.stdout.flush()
                
        return np.mean(total_loss)