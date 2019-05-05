#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn
@CONTACT:zhaojing17@foxmail.com
@SOFTWERE:PyCharm
@FILE:process1.py
@TIME:2019/5/5 16:14
@DES: TensorFlow编程基本流程
'''

if __name__ =="__main__":
    import tensorflow as tf
    import numpy as np

    a = tf.constant(1.,name='const1')
    b = tf.constant(2.,name='const2')
    c = tf.add(a,b)

    with tf.Session() as sess:
        print(sess.run(c))
        print(c.eval)
        # eval() 函数用来执行一个字符串表达式，并返回表达式的值。

    '''运行结果如下：
    3.0
    <bound method Tensor.eval of <tf.Tensor 'Add:0' shape=() dtype=float32>>'''