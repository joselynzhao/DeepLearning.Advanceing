#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn
@CONTACT:zhaojing17@foxmail.com
@SOFTWERE:PyCharm
@FILE:tensorflow_frame.py
@TIME:2019/5/5 17:37
@DES:
'''


if __name__ =="__main__":
    import  sklearn
    import  tensorflow as tf

    # 下面使用TensorFlow的方法

    # ------------------准备训练和测试数据------------------------#
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
    # 随机划分20%的数据作为测试集

    # ------------------placeholder------------------------#

    data = tf.placeholder(tf.float32, [None, 4])
    real_label = tf.placeholder(tf.float32, [None, 1])
    # 给定数据类型和数据大小。None表示本维度根据实际输入数 据自适应调整

    # -------------------定义变量-------------------#

    weight = tf.Variable(tf.random_normal([4, 1]), dtype=tf.float32)
    bias = tf.Variable(tf.ones([1]), dtype=tf.float32)  # 实际使用时只定义 了初值、变量规模和数据类型，默认可训练

    # --------------------损失函数、优化器、优化目标----------------------#

    y_label = tf.add(tf.matmul(data, weight), bias)  # 定 义 回归函数的计算方法
    loss = tf.reduce_mean(tf.square(real_label - y_label))  # 定义目标函数loss
    train = tf.train.GradientDescentOptimizer(0.2).minimize(loss)  # 定义优化器及优化目标(最小化loss), 其中0.2为 学习率

    # ------------------初始化参数------------------------#
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 参 数初始化
        for i in range(1000):  # 训练1000次迭代
            sess.run(train, feed_dict={data: X_train, real_label: y_train})  # 执行训练脚本

        # ------------------配置输入输出及优化器，并启动训练------------------------#
        forecast_set = sess.run(y_label, feed_dict={data: X_test})
        # 执行测试。X_lately: 一部分不包括在 训练集和测试集中的数据，用于生成股价预测结果

    accuracy = tf.reduce_mean(tf.square(forecast_set - y_test))