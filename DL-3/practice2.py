#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn Zhao
@CONTACT:zhaojing17@foxmail.com
@HOME_PAGE:joselynzhao.top
@SOFTWERE:PyCharm
@FILE:practice2.py
@TIME:2019/5/6 16:50
@DES:
'''

import  tensorflow as tf

if __name__ =="__main__":
    with tf.name_scope('name_sp1') as scp1:
        with tf.variable_scope('var_scp2') as scp2:
            with 