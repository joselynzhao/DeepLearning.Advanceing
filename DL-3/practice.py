#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn Zhao
@CONTACT:zhaojing17@foxmail.com
@HOME_PAGE:joselynzhao.top
@SOFTWERE:PyCharm
@FILE:practice.py
@TIME:2019/5/6 16:16
@DES:
'''

import  tensorflow.examples.tutorials.mnist.input_data

import  tensorflow as tf

x = tf.placeholder(tf.float32,[None,784])
W  =tf.Variable(tf.zeros([784,10]))
b  =tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,W)+b)

y_ =tf.placeholder(tf.float32,[None,10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y)))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)



for i in range(1000):
    batch_xs,batch_ys =


correct_prediction = tf.equal(tf.argmax())

