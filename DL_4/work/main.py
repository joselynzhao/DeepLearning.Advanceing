#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn Zhao
@CONTACT:zhaojing17@foxmail.com
@HOME_PAGE:joselynzhao.top
@SOFTWERE:PyCharm
@FILE:main.py
@TIME:2019/5/18 15:54
@DES:
'''

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
import tensorflow.contrib.slim as slim

from tensorflow.python.framework import graph_util

from tensorflow.examples.tutorials.mnist import input_data
from sklearn.utils import shuffle

from lenet import  *

if __name__ =="__main__":
    mnist = input_data.read_data_sets('../../../data/mnist', one_hot=True)
    # X_train = np.pad(mnist.train.images, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    # X_validation = np.pad(mnist.validation.images, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    # X_test = np.pad(mnist.test.images, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    # print("Updated Image Shape: {}".format(X_train[0].shape))
    # X_train, y_train = shuffle(X_train, mnist.train.labels)
    tf.logging.set_verbosity(old_v)
    mylenet = Lenet(0, 0.1, 0.01)
    iteratons = 30000
    batch_size = 64

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for ii in range(iteratons):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            batch_xs = np.pad(batch_xs,((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
            x_test = np.pad(mnist.test.images, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
            sess.run(mylenet.train_step,feed_dict ={mylenet.x:batch_xs,mylenet.y:batch_ys})
            if ii % 500 == 1:
                acc = sess.run(mylenet.accuracy,feed_dict ={mylenet.x:x_test,mylenet.y:mnist.test.labels})
                print("%5d: accuracy is: %4f" % (ii, acc))

        print('[accuracy,loss]:', sess.run([mylenet.accuracy], feed_dict={mylenet.x:x_test,mylenet.y:mnist.test.labels}))







