#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn Zhao
@CONTACT:zhaojing17@foxmail.com
@HOME_PAGE:joselynzhao.top
@SOFTWERE:PyCharm
@FILE:example_code.py
@TIME:2019/5/12 11:11
@DES:
'''

import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

if __name__ =="__main__":
    mnist = read_data_sets("data/",one_hot=True)
    x = tf.placeholder(dtype='float',shape=[None,784])
    w = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x,w)+b)
    y_ = tf.placeholder(dtype='float',shape=[None,10])
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cross_entropy)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    step = 500
    loss_list = []
    for i in range(step):
        batch_xs,batch_ys = mnist.train.next_batch(100) #shape: (100, 784) (100, 10) _,loss,weight= sess.run([train_step,cross_entropy,w],feed_dict={x:batch_xs,y_:batch_ys}) loss_list.append(loss)
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))
    print('[accuracy,loss]:',sess.run([accuracy,cross_entropy],feed_dict={x:mnist.test.images,y_:mnist.test.labels}))