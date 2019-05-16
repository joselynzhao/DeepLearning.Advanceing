#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn Zhao
@CONTACT:zhaojing17@foxmail.com
@HOME_PAGE:joselynzhao.top
@SOFTWERE:PyCharm
@FILE:practice.py
@TIME:2019/5/16 15:31
@DES:
'''
import  tensorflow as tf

def _cnn_layer(self, scope_name, W_name, b_name, x, filter_shape , conv_strides, padding_tag='VALID'):

    with tf.variable_scope(scope_name): #添 加 更 灵 活 的 操 作， 但 构 建 繁琐
        conv_W = tf.get_variable(W_name, dtype=tf.float32, initializer=tf.truncated_normal(shape=filter_shape, mean= self.config.mu, stddev=self.config.sigma))
        conv_b = tf.get_variable(b_name, dtype=tf.float32, initializer=tf.zeros(filter_shape[3]))
        conv = tf.nn.conv2d(x, conv_W, strides=conv_strides, padding=padding_tag) + conv_b
        tf.summary.histogram('weights', conv_W)
        tf.summary.histogram('biases', conv_b)
        tf.summary.histogram('activations', conv)
        return conv