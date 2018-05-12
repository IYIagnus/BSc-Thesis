import numpy as np
import tensorflow as tf
import pandas as pd
from DataProcessing import *

import sys
import time

class Config(object):
    """hyperparameters and data"""
    batch_size = 20
    max_epochs = 2**10
    J = 10
    lr = 0.001
    training_size = 5
    valid_size = 2
    num_runs = 1
    #data = getFormattedData()
    
class RationalityByDesign(object):
    def __init__(self, config):
        self.config = config
        self.W_tilde = tf.Variable(tf.truncated_normal([1, config.J], stddev=0.1))
        self.W_bar = tf.Variable(tf.truncated_normal([1, config.J], stddev=0.1))
        self.W_hat = tf.Variable(tf.truncated_normal([config.J, 1], stddev=0.1))
        self.b_tilde = tf.Variable(tf.zeros(config.J))
        self.b_bar = tf.Variable(tf.zeros(config.J))
        self.add_placeholders()
        self.mse = self.getMSE()
        self.output = self.inference()
        self.loss = self.get_loss()
        self.train_step = self.add_training_op(self.loss)
        self.merged = tf.summary.merge_all()

        
    def add_placeholders(self):
        self.moneyness = tf.placeholder(tf.float32, [self.config.batch_size,], "moneyness")
        self.ttm = tf.placeholder(tf.float32, [self.config.batch_size, 1], "ttm")
        self.r = tf.placeholder(tf.float32, [self.config.batch_size, 1], "r")
        self.S = tf.placeholder(tf.float32, [self.config.batch_size, 1], "S0")
        self.labels = tf.placeholder(tf.float32, [self.config.batch_size,], "labels")
        
    def inference(self):
        with tf.variable_scope("tilde_layer"):
            tilde_layer = tf.nn.softplus(self.b_tilde - tf.matmul(tf.reshape(self.moneyness, [self.config.batch_size, 1]), tf.exp(self.W_tilde)))
        
        with tf.variable_scope("bar_layer"):
            bar_layer = tf.sigmoid(self.b_bar + tf.matmul(tf.reshape(self.ttm, [self.config.batch_size, 1]), tf.exp(self.W_bar)))
        
        with tf.variable_scope("hat_layer"):
            output = tf.matmul(tf.multiply(tilde_layer, bar_layer), self.W_hat)

        return output #tf.multiply(tf.multiply(tf.exp(tf.multiply(-self.r, self.ttm)), self.S), output)
        
    def get_loss(self):
        with tf.variable_scope("loss_function"):
            loss = tf.losses.mean_squared_error(tf.exp(self.r*self.ttm)*tf.reshape(self.labels, [self.config.batch_size, 1])/self.S, self.output)
            #loss = tf.losses.mean_squared_error(tf.reshape(self.labels, [self.config.batch_size, 1]), self.output)
        
            tf.summary.scalar('loss', loss)

        
        return loss
    
    def add_training_op(self, loss):
        """Calculate and apply gradients"""
        with tf.variable_scope("optimizer_and_gradients"):
            opt = tf.train.AdamOptimizer(learning_rate=self.config.lr)
            gvs = opt.compute_gradients(loss)
            
            train_op = opt.apply_gradients(gvs)
            
            return train_op
        
    
    def getMSE(self):
        mse = tf.losses.mean_squared_error(tf.reshape(self.labels, [self.config.batch_size, 1]), tf.exp(-self.r*self.ttm) * self.inference() * self.S)
        
        return mse
    
    def run_epoch(self, session, data, num_epoch=0, train_writer=None, train_op=None):
        config = self.config
        total_steps = len(data) // config.batch_size
        
        if train_op is None:
            train_op = tf.no_op()
            
        total_loss = []
        
        labels = data[:, 5]
        moneyness = data[:, 0]/data[:, 3]
        ttm = data[:, 1]
        r = data[:, 4]
        S0 = data[:, 3]
        
        
        
        p = np.random.permutation(len(labels))
        labels, moneyness, ttm, r, S0 = labels[p], moneyness[p], ttm[p], r[p], S0[p]
    
        #print(data)
        
        for step in range(total_steps):
            index = range(step*config.batch_size,(step+1)*config.batch_size)
            feed_dict = {self.moneyness: moneyness[index].astype(np.float32),
                         self.ttm: ttm[index].astype(np.float32).reshape([config.batch_size, 1]),
                         self.r: r[index].astype(np.float32).reshape([config.batch_size, 1]),
                         self.S: S0[index].astype(np.float32).reshape([config.batch_size, 1]),
                         self.labels: labels[index].astype(np.float32),
                         }
            loss, mse, summary, _ = session.run([self.loss, self.mse, self.merged, train_op], feed_dict=feed_dict)
            
            total_loss.append(mse)
            
            if not(train_writer is None):
                train_writer.add_summary(summary, num_epoch*total_steps + step)
            
            if step % 2 == 0:
                sys.stdout.write('\r{} / {} : loss = {}'.format(
                  step, total_steps, np.mean(total_loss)))
                sys.stdout.flush()
                
        return np.mean(total_loss)