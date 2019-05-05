#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn
@CONTACT:zhaojing17@foxmail.com
@SOFTWERE:PyCharm
@FILE:graph.py
@TIME:2019/5/5 11:56
@DES: 图Graph练习
'''

if __name__ =="__main__":

    # 动态
    # import  torch
    # v1 = torch.rand(1,3)
    # v2 = torch.rand(1,3)
    # sum = v1+v2
    # print(v1)
    # print(v2)
    # print(sum)

    # 静态
    # import  tensorflow as tf
    # v1 = tf.Variable(tf.random_uniform([3]))
    # v2 = tf.Variable(tf.random_uniform([3]))
    # sum2 = tf.add(v1,v2)
    # print(v1)
    # print(v2)
    # print(sum2)

    '''运行结果如下：
    <tf.Variable 'Variable:0' shape=(3,) dtype=float32_ref>
    <tf.Variable 'Variable_1:0' shape=(3,) dtype=float32_ref>
    Tensor("Add:0", shape=(3,), dtype=float32)'''

    # 静态图完整版
    import tensorflow as tf

    v1 = tf.Variable(tf.random_uniform([3]))
    v2 = tf.Variable(tf.random_uniform([3]))
    sum2 = tf.add(v1, v2)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        print(sess.run(v1))
        print(sess.run(v2))
        print(sess.run(sum2))

        '''运行结果：
        [0.6578543  0.625384   0.49183977]
        [0.02306736 0.531626   0.8785937 ]
        [0.6809217 1.15701   1.3704334]'''


