#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn Zhao
@CONTACT:zhaojing17@foxmail.com
@HOME_PAGE:joselynzhao.top
@SOFTWERE:PyCharm
@FILE:train_v2.py
@TIME:2019/6/23 19:26
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

from lenet_v2 import  *

if __name__ =="__main__":
    mnist = input_data.read_data_sets('../../../data/mnist', one_hot=True)
    x_test = np.reshape(mnist.test.images,[-1,28,28,1])
    x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')    # print("Updated Image Shape: {}".format(X_train[0].shape))
    tf.logging.set_verbosity(old_v)

    iteratons = 1000
    batch_size = 8
    ma = 0
    sigma = 0.1
    lr = 0.01
    mylenet = Lenet(ma,sigma,lr)

    image_x = []
    image_y_acc = []
    image_y_loss = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for ii in range(iteratons):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            batch_xs = np.reshape(batch_xs,[-1,28,28,1])
            batch_xs = np.pad(batch_xs,((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
            sess.run(mylenet.train_op,feed_dict ={mylenet.x:batch_xs,mylenet.y_:batch_ys})
            if ii % 10 == 0:
                vali_batch_x,vali_batch_y = mnist.validation.next_batch(100)
                vali_batch_x = np.reshape(vali_batch_x,[-1,28,28,1])
                vali_batch_x = np.pad(vali_batch_x,((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
                acc,loss = sess.run([mylenet.accuracy,mylenet.loss],feed_dict ={mylenet.x:vali_batch_x,mylenet.y_:vali_batch_y})
                print("%5d: accuracy is: %4f , loss is : %4f 。" % (ii, acc, loss))
                image_x.append(ii)
                image_y_acc.append(acc)
                image_y_loss.append(loss)

        plt.plot(image_x, image_y_acc, 'r', label="accuracy")
        plt.plot(image_x, image_y_loss, 'g', label="loss")
        plt.xlabel("iteration")
        plt.ylabel("accuracy")
        plt.title("acc_loss_v2")
        plt.savefig('./save/acc_loss_v2.png')
        plt.show()
        print('[accuracy,loss]:', sess.run([mylenet.accuracy], feed_dict={mylenet.x:x_test,mylenet.y_:mnist.test.labels}))