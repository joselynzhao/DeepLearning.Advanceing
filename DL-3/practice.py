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

import  tensorflow.examples.tutorials.mnist.input_data as input_data

import  tensorflow as tf

if __name__ =="__main__":
    mnist = input_data.read_data_.sets("MNIST_data/", one_hot = True)
    x = tf.placeholder(tf.float32,[None,784])
    W  =tf.Variable(tf.zeros([784,10]))
    b  =tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x,W)+b)

    #我们首先需要添加一个新的占位符用于输入正确值
    y_ =tf.placeholder(tf.float32,[None,10])

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
    # reduction_indices=[1] 这个是什么意思？
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    for i in range(1000):
        batch_xs,batch_ys = mnist.train.next_batch(100)
        sess.run(train_step,feed_dict ={x:batch_xs,y_:batch_ys})



    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
    print sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels})

