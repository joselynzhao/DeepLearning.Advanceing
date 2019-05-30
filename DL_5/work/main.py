#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn Zhao
@CONTACT:zhaojing17@foxmail.com
@HOME_PAGE:joselynzhao.top
@SOFTWERE:PyCharm
@FILE:main.py
@TIME:2019/5/30 21:32
@DES:
'''

from lenet import lenet
import  tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import  numpy as np
import tensorflow.contrib.slim.nets as nets

if __name__ =="__main__":
    mnist = input_data.read_data_sets('../../../data/mnist', one_hot=True)
    x_test = np.reshape(mnist.test.images, [-1, 28, 28, 1])
    x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)),'constant')  # print("Updated Image Shape: {}".format(X_train[0].shape))

    iteratons = 30000
    batch_size = 64
    ma = 0
    sigma = 0.1
    lr = 0.01

    x = tf.placeholder(tf.float32, [None, 32, 32, 1])
    y_ = tf.placeholder(tf.float32, [None, 10])
    y = lenet(x)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.005).minimize(cross_entropy)
    # 准确率
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)
    for i in range(iteratons):
        batch_xs,batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = np.reshape(batch_xs, [-1, 28, 28, 1])
        batch_xs = np.pad(batch_xs, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
        sess.run([train_step,cross_entropy],feed_dict={x:batch_xs,y_:batch_ys})
        if i % 500 ==1:
            acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels})
            print("%5d: accuracy is: %4f" % (i, acc))

    print('[accuracy,loss]:',sess.run([accuracy,cross_entropy],feed_dict={x:mnist.test.images,y_:mnist.test.labels}))
