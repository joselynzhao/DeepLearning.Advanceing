#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn Zhao
@CONTACT:zhaojing17@foxmail.com
@HOME_PAGE:joselynzhao.top
@SOFTWERE:PyCharm
@FILE:example1.py
@TIME:2019/5/13 22:23
@DES:案例:某同学想在学习过程中测试模型(model)，但训练和测试 的输入是不同的(training data、validation data)，batchsize 不 一样，他想这么做
'''

import tensorflow as tf

if __name__ =="__main__":
    X=tf.placeholder(tf.float32)
    def model(X):
        w=tf.Variable(name="w", initial_value=tf. random_normal(shape=[1]))
        m=tf.multiply(X,w)
        return m
    def train_graph(X):
        m=model(X)
        a=tf.add(m,X)
        return a
    def test_graph(X):
        m=model(X)
        b=tf.add(m,X)
        return b
    a=train_graph(X)
    b=test_graph(X)

    X_in = 1.2
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ar = sess.run(a, feed_dict={X:X_in})
        br = sess.run(b, feed_dict={X:X_in})
        print("ar=", ar)
        print("br=", br)