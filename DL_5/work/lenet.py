#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn Zhao
@CONTACT:zhaojing17@foxmail.com
@HOME_PAGE:joselynzhao.top
@SOFTWERE:PyCharm
@FILE:lenet.py
@TIME:2019/5/30 20:47
@DES:
'''

import tensorflow.contrib.slim as slim
import  tensorflow as tf

def lenet(image):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0,0.1),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
        net = slim.conv2d(image, 6, [5, 5], stride=1, padding="VALID", scope="conv1")
        net = slim.max_pool2d(net, [2, 2], stride=2, padding="VALID", scope="pool1")
        net = slim.conv2d(net,16,[5,5],stride=1,padding = "VALID",scope ="conv2")
        net = slim.max_pool2d(net,[2,2],stride=2,padding="VALID",scope="pool2")
        net = slim.flatten(net,scope="flatten")
        net = slim.fully_connected(net,120, scope='fc1')
        net = slim.fully_connected(net,84, scope='fc2')
        net = slim.fully_connected(net,10,activation_fn=None, scope='fc3')
        return net





