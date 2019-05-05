#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn
@CONTACT:zhaojing17@foxmail.com
@SOFTWERE:PyCharm
@FILE:main1.py
@TIME:2019/5/5 20:14
@DES:  函数拟合及两种模式的模型保存程序
'''

import matplotlib.pyplot as plt
import numpy as np
import math
import tensorflow as tf



if __name__  =="__main__":
    # import tensorflow.contrib.eager as tfe
    # tfe.enable_eager_execution()
    school_number = 18023032
    a =18
    b =32
    x = tf.linspace(0.,2*math.pi,2000)
    y = tf.cos(a*x+b)
    # print(x)
    # print(y)
    # print(tf.shape(x))
    # np.reshape(x,[1,2000])
    # np.reshape(y,[1,2000])
    # print(tf.shape(x))
    # print(np.shape(y))
    # tf.reshape(y,[1,2000])

    # plt.title("test")
    # plt.plot(x,y)
    # plt.show()


    data = tf.placeholder(tf.float32,[1])
    label = tf.placeholder(tf.float32,[1])

    print(tf.shape(data))
    print(tf.shape(label))
    # tf.reshape(data,[2000,1])
    # print(tf.shape(data))


    w1 = tf.Variable(tf.random_normal([1]),dtype=tf.float32)
    w2 = tf.Variable(tf.random_normal([1]),dtype=tf.float32)
    w3 = tf.Variable(tf.random_normal([1]),dtype=tf.float32)
    b = tf.Variable(tf.random_normal([1]),dtype=tf.float32)

    y_label = tf.add(tf.add(data*w1,(data**2)*w2),tf.add((data**3)*w3,b))
    # y_label = tf.add(tf.add(tf.matmul(data,w1),tf.matmul(data**2,w2)),tf.add(tf.matmul(data**3,w3),b))
    loss = tf.reduce_mean(tf.square(label - y_label))
    train = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # sess.run(x)
        # sess.run(y)
        # sess.run(tf.reshape(x,[2000,1]))
        # sess.run(tf.reshape(y,[2000,1]))
        # print(x)
        # print(y)
        for i in range(2000):
            sess.run(train, feed_dict={data:x,label:y})
    #



