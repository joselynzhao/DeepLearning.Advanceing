#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn
@CONTACT:zhaojing17@foxmail.com
@SOFTWERE:PyCharm
@FILE:practice1.py
@TIME:2019/5/5 11:38
@DES:
'''

import  tensorflow as tf
import torch



if __name__ == "__main__":

    # def graph_a():
    #     a = tf.constant(2, name='a')
    #     # print(a.name,a)
    #
    #     b = tf.constant(3, name='b')
    #     # print(b.name, b)
    #
    #     result = tf.add(a,b,name = 'add')
    #
    #
    #
    # result = 0
    # sess = tf.Session()
    # xx = sess.run(result)
    # print(result)


    c= tf.constant([1.0, 2, 0], name='c')
    d = tf.constant([2.0, 3, 0], name='d')
    result = c + d
    print result
    # 运行结果为： Tensor("add:0", shape=(3,), dtype=float32) 。并未如ppt所说给出运行结果