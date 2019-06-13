#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn Zhao
@CONTACT:zhaojing17@foxmail.com
@HOME_PAGE:joselynzhao.top
@SOFTWERE:PyCharm
@FILE:main.py
@TIME:2019/6/13 10:32
@DES:
'''


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data
from Lenet import  *

mnist = input_data.read_data_sets('../../../data/mnist', one_hot=True)

iteratons = 100
batch_size = 64
ma = 0
sigma = 0.1
lr = 0.01


def train_lenet(lenet):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        tf.summary.image("input",lenet.x,3)
        merged_summary = tf.summary.merge_all()

        writer = tf.summary.FileWriter("LOGDIR/",sess.graph)
        # writer.add_graph(sess.graph)

        for ii in range(iteratons):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            batch_xs = np.reshape(batch_xs,[-1,28,28,1])
            batch_xs = np.pad(batch_xs,((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
            sess.run(lenet.train_step,feed_dict ={lenet.x:batch_xs,lenet.y_:batch_ys})
            if ii % 50 == 1:
                acc,s = sess.run([lenet.accuracy,merged_summary],feed_dict ={lenet.x:x_test,lenet.y_:mnist.test.labels})
                writer.add_summary(s,ii)
                print("%5d: accuracy is: %4f" % (ii, acc))
        print('[accuracy,loss]:', sess.run([lenet.accuracy], feed_dict={lenet.x:x_test,lenet.y_:mnist.test.labels}))


if __name__ =="__main__":
    for act in ["relu","sigmoid"]:
        x_test = np.reshape(mnist.test.images, [-1, 28, 28, 1])
        x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)),'constant')  # print("Updated Image Shape: {}".format(X_train[0].shape))
        tf.logging.set_verbosity(old_v)
        lenet = Lenet(ma,sigma,lr,act)
        train_lenet(lenet)