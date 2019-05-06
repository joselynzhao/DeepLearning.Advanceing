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
    a =18.0
    b =32.0
    N = 2000
    x = np.linspace(float(-b/a),(2*math.pi-b)/a,N).reshape([-1,1])
    y = np.cos(a*x+b).reshape([-1,1])

    # plt.plot(x,y)
    # plt.show(-


    data = tf.placeholder(tf.float32,[None,1])
    label = tf.placeholder(tf.float32,[None,1])

    print(tf.shape(data))
    print(tf.shape(label))
    # print(tf.shape(data))


    w1 = tf.Variable(tf.random_normal([1,1],mean=0, stddev=200),dtype=tf.float32, name='s_w1')
    w2 = tf.Variable(tf.random_normal([1,1],mean=100, stddev=200),dtype=tf.float32, name='s_w2')
    w3 = tf.Variable(tf.random_normal([1,1],mean=200, stddev=200),dtype=tf.float32, name='s_w3')
    b = tf.Variable(tf.random_normal([1,1],mean=200, stddev=100),dtype=tf.float32, name='s_b')

    # 200 100   0.35
    # 200 10    0.36
    # 300 200 100 50 0.34995785
    # 500 100 50 100 0.33685145

    y_label = tf.add(tf.add(tf.matmul(data,w1),tf.matmul((data**2),w2)),tf.add(tf.matmul((data**3),w3),b))
    loss = tf.reduce_mean(tf.square(label - y_label))
    train = tf.train.AdamOptimizer(0.2).minimize(loss)
    # train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    # train = tf.train.AdadeltaOptimizer(0.1).minimize(loss)
    # train = tf.train.AdagradOptimizer(0.1).minimize(loss)
    # train = tf.train.FtrlOptimizer(0.01).minimize(loss)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(50000):
            sess.run(train, feed_dict={data:x,label:y})
            if i % 500 == 0:
                log_loss = sess.run(loss,feed_dict={data:x,label:y})
                print(i,log_loss)
        print(sess.run(w1))
        print(sess.run(w2))
        print(sess.run(w3))
        print(sess.run(b))

        # ckpt 保存
        save_path1 = saver.save(sess,save_path_ckpt)

        # pb保存
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['s_w1','s_w2','s_w3','s_b'])
        with tf.gfile.FastGFile(save_path_pb, mode='wb') as f:
            f.write(constant_graph.SerializeToString())






