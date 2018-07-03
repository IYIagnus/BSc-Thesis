import numpy as np
import tensorflow as tf
import pandas as pd
from DataProcessing import *

import sys
import time

class Config(object):
    """hyperparameters and data"""
    batch_size = 32
    max_epochs = 2**10
    J = 5
    K = 5
    I = 9
    P = 5
    lr = 0.001
    hint_lr = 0.0004
    training_size = 5
    test_size = 1
    num_runs = 3
    dropout = 1
    l2 = 0
    delta = 0.001
    mode = None
    early_stopping = 30
    #data = getFormattedData()
    
def standardizeInput(variable):
        mean = variable.mean()
        stddev = variable.std()
        
        standardized_variable = (variable - mean) / stddev
                
        return standardized_variable
    
class RationalityByDesign(object):
    def __init__(self, config):
        self.config = config
        self.add_placeholders()
        with tf.variable_scope("output", reuse=tf.AUTO_REUSE):
            self.output = self.inference(self.moneyness)
            self.delta_output = self.inference(self.delta_moneyness) #learning from hints
        self.loss = self.get_loss()
        self.hint = self.get_hint_loss()
        self.train_step = self.add_training_op(self.loss)
        self.hint_step = self.add_hint_op(self.hint)
        self.merged = tf.summary.merge_all()
        self.mse = self.getMSE()

        
    def add_placeholders(self):
        self.moneyness = tf.placeholder(tf.float32, [self.config.batch_size,], "moneyness")
        self.delta_moneyness = tf.placeholder(tf.float32, [self.config.batch_size,], "delta_moneyness")
        self.stdttm = tf.placeholder(tf.float32, [self.config.batch_size, 1], "stdttm")
        self.ttm = tf.placeholder(tf.float32, [self.config.batch_size, 1], "ttm")
        self.r = tf.placeholder(tf.float32, [self.config.batch_size, 1], "r")
        self.S = tf.placeholder(tf.float32, [self.config.batch_size, 1], "S0")
        self.labels = tf.placeholder(tf.float32, [self.config.batch_size,], "labels")
    
    def single_model(self, moneyness):
        #init weights
        w_tilde = tf.get_variable("w_tilde", [1, self.config.J], initializer=tf.truncated_normal_initializer(stddev=0.1)) #tf.Variable(tf.truncated_normal([1, self.config.J], stddev=0.1))
        w_bar = tf.get_variable("w_bar", [1, self.config.J], initializer=tf.truncated_normal_initializer(stddev=0.1))#tf.Variable(tf.truncated_normal([1, self.config.J], stddev=0.1))
        w_hat = tf.get_variable("w_hat", [self.config.J, 1], initializer=tf.truncated_normal_initializer(stddev=0.1))#tf.Variable(tf.truncated_normal([self.config.J, 1], stddev=0.1))
        
        #init biases
        b_tilde = tf.get_variable("b_tilde", self.config.J, initializer=tf.constant_initializer(0.0))#tf.Variable(tf.zeros(self.config.J), name='bias_tilde')
        b_bar = tf.get_variable("b_bar", self.config.J, initializer=tf.constant_initializer(0.0))#tf.Variable(tf.zeros(self.config.J), name='bias_bar')
        
        with tf.variable_scope("tilde_layer"):
            tilde_layer = tf.nn.softplus(b_tilde - tf.matmul(tf.reshape(moneyness, [self.config.batch_size, 1]), tf.exp(w_tilde)))
        
        with tf.variable_scope("bar_layer"):
            bar_layer = tf.sigmoid(b_bar + tf.matmul(self.stdttm, tf.exp(w_bar)))
         
        if self.config.mode == "train":
            with tf.variable_scope("dropout"):
                multiply = tf.nn.dropout(tf.multiply(tilde_layer, bar_layer), self.config.dropout)
        else:
            multiply = tf.multiply(tilde_layer, bar_layer)
        
        with tf.variable_scope("hat_layer"):
            output = tf.matmul(multiply, w_hat)

        return output #tf.multiply(tf.multiply(tf.exp(tf.multiply(-self.r, self.ttm)), self.S), output) #shape (batch_size x 1)
    
    def weight_model(self, moneyness):
        #init weights
        W_dot = tf.get_variable("w_dot", [2, self.config.K], initializer=tf.truncated_normal_initializer(stddev=1))#tf.Variable(tf.truncated_normal([2, self.config.K], stddev=1))
        W_umlaut = tf.get_variable("w_umlaut", [self.config.K, self.config.I], initializer=tf.truncated_normal_initializer(stddev=1))#tf.Variable(tf.truncated_normal([self.config.K, self.config.I], stddev=1))
        
        #init biases
        b_dot = tf.get_variable("b_dot", self.config.K, initializer=tf.constant_initializer(0.0))#tf.Variable(tf.zeros(self.config.K), name='bias_dot')
        b_umlaut = tf.get_variable("b_umlaut", self.config.I, initializer=tf.constant_initializer(0.0))#tf.Variable(tf.zeros(self.config.I), name='bias_umlaut')
        
        with tf.variable_scope("dot_layer"):
            dot_layer = tf.sigmoid(b_dot + tf.matmul(tf.concat([tf.reshape(moneyness, [self.config.batch_size, 1]), self.stdttm], 1), W_dot)) # shape (batch_size x K)
        
        with tf.variable_scope("umlaut_layer"):
            umlaut_layer = b_umlaut + tf.matmul(dot_layer, W_umlaut)
            exp_umlaut = tf.exp(umlaut_layer) #shape (batch_size x I)
            
        with tf.variable_scope("weighted_mean"):
            output = exp_umlaut/tf.reduce_sum(exp_umlaut, 1, keep_dims=True)
        
        if self.config.mode == "train":
            output = tf.nn.dropout(output, self.config.dropout)
        
        return output #shape (batch_size x I)
        
    def inference(self, moneyness):
        with tf.variable_scope("inference"):   
            with tf.variable_scope("vectorize_ys"):
                y = []
                for i in range(self.config.I):
                    with tf.variable_scope(str(i)):
                        y.append(self.single_model(moneyness))
                
                y = tf.reshape(tf.stack(y, axis=1), [self.config.batch_size, self.config.I]) #shape (batch_size x I)
            
            with tf.variable_scope("get_weights"):
                w = self.weight_model(moneyness) #shape (batch_size x I)
                
            with tf.variable_scope("hadamard_product"):
                product = tf.multiply(y, w) #shape (batch_size x I)
        
            output = tf.reduce_sum(product, 1, keep_dims=True)
            
        return output
        
    def get_loss(self):
        labels = tf.exp(self.r*self.ttm)*tf.reshape(self.labels, [self.config.batch_size, 1])/self.S
        with tf.variable_scope("compute_loss"):
            with tf.variable_scope("mse_loss"):
                mse = tf.losses.mean_squared_error(labels, self.output)
                #loss = tf.losses.mean_squared_error(tf.reshape(self.labels, [self.config.batch_size, 1]), self.output)
            
            with tf.variable_scope("mape_loss"):
                mape = (100/self.config.batch_size)*tf.reduce_sum(((labels-self.output)/labels))
                
            loss = mse
            
            with tf.variable_scope("L2_loss"):
                for v in tf.trainable_variables():
                        if not 'bias' in v.name.lower():
                            loss += self.config.l2*tf.nn.l2_loss(v)
                
            tf.summary.scalar('loss', loss)
        
        return loss
    
    def get_hint_loss(self):
        with tf.variable_scope("hint_loss"):
                g = tf.gradients(self.output, self.moneyness)
                g_delta = tf.gradients(self.delta_output, self.delta_moneyness)
                
                delta_loss = tf.maximum(float(0), g[0]-g_delta[0])
                loss = tf.reduce_sum(delta_loss)
                
        return loss
    
    def add_training_op(self, loss):
        """Calculate and apply gradients"""
        with tf.variable_scope("optimizer_and_gradients"):
            opt = tf.train.AdamOptimizer(learning_rate=self.config.lr)
            gvs = opt.compute_gradients(loss)
            
            train_op = opt.apply_gradients(gvs)
            
            return train_op
        
    def add_hint_op(self, loss):
        with tf.variable_scope("hint_optimizer"):
            opt = tf.train.AdamOptimizer(learning_rate=self.config.hint_lr)
            gvs = opt.compute_gradients(loss)
            
            hint_op = opt.apply_gradients(gvs)
            
            return hint_op
        
    def getMSE(self):
        mse = tf.losses.mean_squared_error(tf.reshape(self.labels, [self.config.batch_size, 1]),
                                           tf.exp(-self.r*self.ttm)*self.output*self.S)
        
        return mse
    
    def run_epoch(self, session, data, mode=None, num_epoch=0, train_writer=None, train_op=None):
        """input:
            data = numpy array of shape (n x 6), i.e. n samples of [strike_price, ttm, dividend_yield, S0, r, call_price]"""
        if mode == "train":
            self.config.mode = "train"
        elif mode == "hint":
            self.config.mode = "hint"
        else:
            self.config.mode = None
            
        config = self.config
        total_steps = len(data) // config.batch_size
        
        if train_op is None:
            train_op = tf.no_op()
            
        total_loss = []
        
        moneyness = data[:, 0]/data[:, 3]
        ttm = data[:, 1]
        r = data[:, 4]/100
        S0 = data[:, 3]
        labels = data[:, 5]
                
        moneyness = standardizeInput(moneyness)
        stdttm = standardizeInput(ttm)

        if mode is not("test"):
            p = np.random.permutation(len(labels))
            labels, moneyness, stdttm, ttm, r, S0 = labels[p], moneyness[p], stdttm[p], ttm[p], r[p], S0[p]
    
        #print(data)
        
        for step in range(total_steps):
            index = range(step*config.batch_size,(step+1)*config.batch_size)
            feed_dict = {self.moneyness: moneyness[index].astype(np.float32),
                         self.delta_moneyness: moneyness[index].astype(np.float32) + config.delta,
                         self.stdttm: stdttm[index].astype(np.float32).reshape([config.batch_size, 1]),
                         self.ttm: ttm[index].astype(np.float32).reshape([config.batch_size, 1]),
                         self.r: r[index].astype(np.float32).reshape([config.batch_size, 1]),
                         self.S: S0[index].astype(np.float32).reshape([config.batch_size, 1]),
                         self.labels: labels[index].astype(np.float32),
                         }
            if mode == "hint":
                loss, summary, _ = session.run([self.hint, self.merged, train_op], feed_dict=feed_dict)
                total_loss.append(loss)
            elif mode == "test":
                mse, output, _ = session.run([self.mse, tf.exp(-self.r*self.ttm)*self.output*self.S, train_op], feed_dict=feed_dict)
                total_loss.append(mse)
            else:
                loss, mse, summary, _ = session.run([self.loss, self.mse, self.merged, train_op], feed_dict=feed_dict)
                total_loss.append(mse)
            
            if not(train_writer is None):
                train_writer.add_summary(summary, num_epoch*total_steps + step)
        if mode is not("test"):        
            return np.mean(total_loss)
        else:
            return np.mean(total_loss), output