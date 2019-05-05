#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn
@CONTACT:zhaojing17@foxmail.com
@SOFTWERE:PyCharm
@FILE:eager_execution.py
@TIME:2019/5/5 19:30
@DES:
'''

if __name__ =="__main__":
    import  tensorflow as tf
    import  tensorflow.contrib.eager as tfe
    tfe.enable_eager_execution()
    x = [[2.]]
    m = tf.matmul(x,x)

    print(m)
    '''tf.Tensor([[4.]], shape=(1, 1), dtype=float32)'''