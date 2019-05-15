#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn Zhao
@CONTACT:zhaojing17@foxmail.com
@HOME_PAGE:joselynzhao.top
@SOFTWERE:PyCharm
@FILE:main2.py
@TIME:2019/5/14 19:06
@DES:
'''

import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib
import tensorflow as tf
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from tensorflow.examples.tutorials.mnist import input_data

if __name__ =="__main__":

    mnist = input_data.read_data_sets('./data/mnist', one_hot=True)
    tf.logging.set_verbosity(old_v)
    def X_W(x,reuse=False):
        with tf.variable_scope("X_W") as scope:
            if reuse:
                scope.reuse_variables()
            W = tf.Variable(tf.zeros([392,10]),name='w')
            y = tf.matmul(x,W)
            return W,y

    input1 = tf.placeholder(dtype='float',shape=[None,392])
    input2 = tf.placeholder(dtype='float',shape=[None,392])

    # x = tf.placeholder(dtype='float',shape=[None,784])
    # w = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))
    _,y1 = X_W(input1)
    weight,y2 = X_W(input2,True)
    y = tf.nn.softmax(y1+y2+b)
    y_ = tf.placeholder(dtype='float',shape=[None,10])

    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.005).minimize(cross_entropy)
    #准确率
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    step = 10000
    batch_size = 64
    # loss_list = []
    for i in range(step):
        batch_xs,batch_ys = mnist.train.next_batch(batch_size)
        _,loss,w= sess.run([train_step,cross_entropy,weight],feed_dict={input1:batch_xs[:,0:392],input2:batch_xs[:,392:784],y_:batch_ys})
        if i % 500 ==1:
            acc = sess.run(accuracy,feed_dict={input1:mnist.test.images[:,0:392],input2:mnist.test.images[:,392:784],y_:mnist.test.labels})
            print("%5d: accuracy is: %4f" % (i, acc))

    print('[accuracy,loss]:',sess.run([accuracy,cross_entropy],feed_dict={input1:mnist.test.images[:, 0:392],input2:mnist.test.images[:, 392:784],y_:mnist.test.labels}))
    w = np.array(w)
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 7,}
    matplotlib.rc('font', **font1)
    plt.figure()
    for i in range(10):
        weight = w[:,i]
        weight = np.reshape(weight,[14,28])
        plt.subplot(5,2,i)
        plt.title(i)
        plt.imshow(weight)
    plt.savefig('./save/weight.png')
    plt.show()
