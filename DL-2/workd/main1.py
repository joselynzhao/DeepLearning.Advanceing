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
from tensorflow.python.framework import graph_util

save_path_ckpt = './save/dl-2-work.ckpt'
save_path_pb = './save/dl-2-work.pb'



if __name__  =="__main__":
    # import tensorflow.contrib.eager as tfe
    # tfe.enable_eager_execution()
    school_number = 18023032
    a =18
    b =32
    x = np.linspace(0.,2*math.pi,2000).reshape([-1,1])
    y = np.cos(a*x+b).reshape([-1,1])
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


    data = tf.placeholder(tf.float32,[None,1])
    label = tf.placeholder(tf.float32,[None,1])

    print(tf.shape(data))
    print(tf.shape(label))
    # tf.reshape(data,[2000,1])
    # print(tf.shape(data))


    w1 = tf.Variable(tf.random_normal([1]),dtype=tf.float32,name='op_to_store')
    w2 = tf.Variable(tf.random_normal([1]),dtype=tf.float32,name='op_to_store')
    w3 = tf.Variable(tf.random_normal([1]),dtype=tf.float32,name='op_to_store')
    b = tf.Variable(tf.random_normal([1]),dtype=tf.float32,name='op_to_store')

    y_label = tf.add(tf.add(data*w1,(data**2)*w2),tf.add((data**3)*w3,b))
    # y_label = tf.add(tf.add(tf.matmul(data,w1),tf.matmul(data**2,w2)),tf.add(tf.matmul(data**3,w3),b))
    loss = tf.reduce_mean(tf.square(label - y_label))
    train = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # sess.run(x)
        # sess.run(y)
        # sess.run(tf.reshape(x,[2000,1]))
        # sess.run(tf.reshape(y,[2000,1]))
        # print(x)
        # print(y)
        for i in range(300000):
            sess.run(train, feed_dict={data:x,label:y})
            if i% 100 == 1:
                print(sess.run(loss,feed_dict={data:x,label:y}))
        print(sess.run(w1))
        print(sess.run(w2))
        print(sess.run(w3))
        print(sess.run(b))
        save_path1 = saver.save(sess,save_path_ckpt)

        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['op_to_store'])
        with tf.gfile.FastGFile(save_path_pb, mode='wb') as f:
            f.write(constant_graph.SerializeToString())


    #



