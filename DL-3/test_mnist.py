#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn Zhao
@CONTACT:zhaojing17@foxmail.com
@HOME_PAGE:joselynzhao.top
@SOFTWERE:PyCharm
@FILE:example_code.py
@TIME:2019/5/12 11:11
@DES:
'''


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow.python.framework import graph_util

from tensorflow.examples.tutorials.mnist import input_data


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


# 动态图绘制器
class gif_drawer():
    def __init__(self):
        plt.ion()
        self.xs = [0, 0]
        self.ys = [0, 0]

    def draw(self, update_x, update_y):
        self.xs[0] = self.xs[1]
        self.ys[0] = self.ys[1]

        self.xs[1] = update_x
        self.ys[1] = update_y

        plt.title("Training Accuracy")
        plt.xlabel("iteration")
        plt.ylabel("accuracy")
        plt.plot(self.xs, self.ys, )
        plt.pause(0.1)


# 封装的MNIST预测器
class FC_MNIST():
    def __init__(self, sess, iterations=20000, lr=0.02, batch_size=64):
        self.sess = sess
        # 定义超参数
        self.iterations = iterations
        self.lr = lr
        self.batch_size = batch_size
        # 初始化绘制器
        self.gd = gif_drawer()
        # 建立模型
        self.build_model()

    def build_model(self):
        # 定义占位符
        self.x1 = tf.placeholder(tf.float32, [None, 392])
        self.x2 = tf.placeholder(tf.float32, [None, 392])
        self.y_ = tf.placeholder(tf.float32, [None, 10])
        # 调用网络
        y = self.FCC(self.x1, self.x2)
        # 定义损失函数
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=self.y_))
        # 定义优化器
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.cross_entropy)

        # 定义正确率
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def _X_W(self, X, reuse=False):
        with tf.variable_scope("X_W") as scope:
            if reuse:
                scope.reuse_variables()
            self.W = tf.Variable(tf.zeros([392, 10]), name='w')
            return tf.matmul(X, self.W)

    # 定义网络模型
    def FCC(self, x1, x2):
        self.b = tf.Variable(tf.zeros([10]))
        y = tf.nn.softmax(self._X_W(x1) + self._X_W(x2, True) + self.b)
        return y

    # 定义训练函数
    def train(self):
        tf.global_variables_initializer().run()
        for ii in range(self.iterations):
            batch_xs, batch_ys = mnist.train.next_batch(self.batch_size)
            self.sess.run(self.train_step, feed_dict={self.x1: batch_xs[:, 0:392], \
                                                      self.x2: batch_xs[:, 392:784], self.y_: batch_ys})
            if ii % 500 == 1:
                acc, los = self.sess.run([self.accuracy, self.cross_entropy], \
                                         feed_dict={self.x1: mnist.test.images[:, 0:392], \
                                                    self.x2: mnist.test.images[:, 392:784], \
                                                    self.y_: mnist.test.labels})
                print("Iteration [%5d/%5d]: accuracy is: %4f loss is: %4f" % (ii, self.iterations, acc, los))
                self.gd.draw(ii, acc)

    # 定义测试函数
    def test(self):
        acc = self.sess.run(self.accuracy, \
                            feed_dict={self.x1: mnist.test.images[:, 0:392], \
                                       self.x2: mnist.test.images[:, 392:784], \
                                       self.y_: mnist.test.labels})
        print("Test: accuracy is %4f" % (acc))


if __name__ == '__main__':
    mnist = input_data.read_data_sets('./data/mnist', one_hot=True)
    # 新建会话
    with tf.Session() as sess:
        # 实例化对象
        fcc = FC_MNIST(sess)
        # 启动训练
        fcc.train()
        # 测试
        fcc.test()