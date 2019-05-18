#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn Zhao
@CONTACT:zhaojing17@foxmail.com
@HOME_PAGE:joselynzhao.top
@SOFTWERE:PyCharm
@FILE:lenet1.py
@TIME:2019/5/16 16:17
@DES:
'''

import tensorflow as tf
from example_jicheng import *


class Lenet(Network):
    #Network 父类，表示Lenet继承于Network
    # __init__类 似C++构 造 函 数， 实 例 化 创 建 对 象 时 调 用， 做 对 象 的 一 些 初始化工作
    def __init__(self, config):
        self.config = config
        self._build_graph()
        #没 有 前 缀 “_",公 有;"__",私 有;"_",保 护
        # #涉及网络的所有画图build graph过程，常用一个build graph封起来
    def _build_graph(self, network_name='Lenet'):
        self._setup_placeholders_graph()
        self._build_network_graph(network_name)
        self._compute_loss_graph()
        self._compute_acc_graph()
        self._create_train_op_graph()
        self.merged_summary = tf.summary.merge_all()

    def _setup_placeholders_graph(self):
        self.x = tf.placeholder("float", shape=[None, 32, 32, 1], name='x')
        self.y_ = tf.placeholder("float", shape=[None, 10], name='y_ ')
        self.keep_prob = tf.placeholder("float", name='keep_prob')

    def _cnn_layer(self, scope_name, W_name, b_name, x, filter_shape, conv_strides, padding_tag='VALID'):
        pass
    def _pooling_layer(self, scope_name, x, pool_ksize,pool_strides, padding_tag='VALID'):
        pass
    def _fully_connected_layer(self, scope_name, W_name, b_name, x, W_shape):
        pass


    def _build_network_graph(self, scope_name):
        with tf.variable_scope(scope_name):
            conv1 = self._cnn_layer('layer_1_conv', 'conv1_w', ' conv1_b', self.x, (5, 5, 1, 6), [1, 1, 1, 1])
            self.conv1 = tf.nn.relu(conv1)
            self.pool1 = self._pooling_layer('layer_1_pooling', self.conv1, [1, 2, 2, 1], [1, 2, 2, 1])
            self.y_predicted = tf.nn.softmax(self.logits)
            tf.summary.histogram("y_predicted", self.y_predicted)

    def _compute_loss_graph(self):
        with tf.name_scope("loss_function"):
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.logits)
            self.loss = tf.reduce_mean(loss)
            tf.summary.scalar("cross_entropy", self.loss)



