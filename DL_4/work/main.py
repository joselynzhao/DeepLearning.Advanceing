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

from lenet import  *

if __name__ =="__main__":
    mnist = input_data.read_data_sets('../../../data/mnist', one_hot=True)
    tf.logging.set_verbosity(old_v)
    mylenet = Lenet(0, 0.1, 0.01)
    iteratons = 30000
    batch_size = 64

    with tf.Seesion() as sess:
        sess.run(tf.global_variables_initializer())
        for ii in range(iteratons):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(mylenet.train_step,feed_dict ={mylenet.x:batch_xs,mylenet.y:batch_ys})
            if ii % 500 == 1:
                acc = sess.run(mylenet.accuracy,feed_dict ={mylenet.x:mnist.test.images,mylenet.y:mnist.test.labels})
                print("%5d: accuracy is: %4f" % (ii, acc))

        print('[accuracy,loss]:', sess.run([mylenet.accuracy], feed_dict={mylenet.x:mnist.test.images,mylenet.y:mnist.test.labels}))







