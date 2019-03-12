# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 17:48:36 2018

@author: Zach
"""

import random
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_input = 784
n_hidden1 = 512
n_hidden2 = 256
n_output = 10

beta = 0.001
start_learning_rate = 0.001
training_epochs = 10
batch_size = 128

#Placeholders for data and labels
X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None, n_output])
keep_prob = tf.placeholder(tf.float32)

#Weights
W1 = tf.Variable(tf.random_normal([n_input, n_hidden1], stddev = 0.01))
W2 = tf.Variable(tf.random_normal([n_hidden1, n_hidden2], stddev = 0.01))
W3 = tf.Variable(tf.random_normal([n_hidden2, n_output], stddev = 0.01))

#Bias
b1 = tf.Variable(tf.random_normal([n_hidden1], stddev = 0.01))
b2 = tf.Variable(tf.random_normal([n_hidden2], stddev = 0.01))
b3 = tf.Variable(tf.random_normal([n_output], stddev = 0.01))
    
#Model
hidden1_layer = tf.nn.relu(tf.add(tf.matmul(X, W1) + b1))
hidden1_dropout = tf.nn.dropout(hidden1_layer, keep_prob)
hidden2_layer = tf.nn.relu(tf.matmul(hidden1_layer, W2) + b2)
hidden2_dropout = tf.nn.dropout(hidden2_layer, keep_prob)
output_layer = tf.sigmoid(tf.matmul(hidden2_dropout, W3) + b3)

#Loss & L2 Regularization
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = output_layer, labels = Y))
regularizer = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3)
loss = tf.reduce_mean(loss + beta * regularizer)

#Optimizer
global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, 100000, 0.96, staircase=True)
train_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

#Metrics
correct_prediction = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as session :
    tf.global_variables_initializer().run()
    for epoch in range(epochs):
        n_batches = int(mnist.train.images.shape[0] / batch_size)
        for batch in range(n_batches):
            batch_data = mnist.train.images[batch_size * batch: batch_size * (batch + 1)]
            batch_labels = mnist.train.labels[batch_size * batch: batch_size * (batch + 1)]
            session.run(train_optimizer, feed_dict = {X: batch_data, Y: batch_labels, keep_prob: 0.6})
        accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))
session.close()
    

    

        
        
        



