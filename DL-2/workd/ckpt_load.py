#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn
@CONTACT:zhaojing17@foxmail.com
@SOFTWERE:PyCharm
@FILE:ckpt_load.py
@TIME:2019/5/6 09:40
@DES:  ckpt模型恢复程序
'''

import  tensorflow as tf
import  numpy as np
import  math
import matplotlib.pyplot as plt

save_path_ckpt = './save/dl-2-work.ckpt'

if  __name__ == "__main__":
    data = tf.placeholder(tf.float32, [None, 1])
    label = tf.placeholder(tf.float32, [None, 1])

    print(tf.shape(data))
    print(tf.shape(label))
    # print(tf.shape(data))

    w1 = tf.Variable(tf.random_normal([1, 1]), dtype=tf.float32, name='s_w1')
    w2 = tf.Variable(tf.random_normal([1, 1]), dtype=tf.float32, name='s_w2')
    w3 = tf.Variable(tf.random_normal([1, 1]), dtype=tf.float32, name='s_w3')
    b = tf.Variable(tf.random_normal([1, 1]), dtype=tf.float32, name='s_b')

    y_label = tf.add(tf.add(tf.matmul(data, w1), tf.matmul((data ** 2), w2)), tf.add(tf.matmul((data ** 3), w3), b))
    loss = tf.reduce_mean(tf.square(label - y_label))
    train = tf.train.AdamOptimizer(0.2).minimize(loss)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess,save_path_ckpt)
        print(sess.run(w1))
        print(sess.run(w2))
        print(sess.run(w3))
        print(sess.run(b))
        w1 = w1.eval()
        w2 = w2.eval()
        w3 = w3.eval()
        b = b.eval()

    school_number = 18023032
    aa = 18
    bb = 32
    N = 2000
    x1 = np.linspace(-bb/aa,(2*math.pi-bb)/aa,N)
    y1 = np.cos(aa * x1 + bb)
    y2 = x1*w1+(x1**2)*w2+(x1**3)*w3+b
    plt.plot(x1,y1,'r')
    plt.plot(x1,y2,'g')
    plt.title("test")
    plt.show()



