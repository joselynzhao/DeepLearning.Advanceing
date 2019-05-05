#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn
@CONTACT:zhaojing17@foxmail.com
@SOFTWERE:PyCharm
@FILE:random_number1.py
@TIME:2019/5/5 16:20
@DES: 示例：随机数生成
'''


if __name__ =="__main__":
    import  numpy as np
    a  = np.random.rand(1)
    for i in range(5):
        print(a)

    '''result:
    [0.99835465]
    [0.99835465]
    [0.99835465]
    [0.99835465]
    [0.99835465]
    '''
    '''即生成的五个随机数是一样的'''

    import  tensorflow as tf
    import  numpy as np

    a = tf.random_normal([1],name = "random")
    with tf.Session() as sess:
        for i in range(5):
            print(sess.run(a))

    '''result:
    [-0.28919014]
    [-0.516945]
    [-0.5970153]
    [1.6492158]
    [0.2942117]'''
    '''五个随机数各不相同'''