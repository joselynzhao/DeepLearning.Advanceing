#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn Zhao
@CONTACT:zhaojing17@foxmail.com
@HOME_PAGE:joselynzhao.top
@SOFTWERE:PyCharm
@FILE:example2.py
@TIME:2019/5/14 10:00
@DES:
'''

import tensorflow as tf
import  numpy as np

if __name__ =="__main__":
    var = tf.Variable(0.,name='var')
    const = tf.constant(1.)
    add_op = tf.add(var,const,name='myAdd')

    assign_op = tf.assign(var, add_op, name='myAssign')

    out1 = assign_op*1
    out2 = assign_op*2

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(3):
            print "var:",sess.run(var),sess.run(out1),sess.run(out2)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(3):
            print "var:",sess.run(var),sess.run([out1,out2])