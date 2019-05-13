#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn Zhao
@CONTACT:zhaojing17@foxmail.com
@HOME_PAGE:joselynzhao.top
@SOFTWERE:PyCharm
@FILE:practice3.py
@TIME:2019/5/13 22:10
@DES:
'''
import  tensorflow as tf

def rnn(inputs,state,hidden_size):
    in_x = tf.concat([inputs,state],axis =1)
    W_shape = [int(in_x.get_shape()[1]),hidden_size]
    b_shape = [1,hidden_size]

    W = tf.get_variable(shape = W_shape, name="weight")
    b = tf.get_variable(shape = b_shape,name = "bias")

    out_linear = tf.nn.bias_add(tf.matmul(in_x,W),b)
    output = tf.nn.tanh(out_linear)
    return output

