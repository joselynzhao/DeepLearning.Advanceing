#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn Zhao
@CONTACT:zhaojing17@foxmail.com
@HOME_PAGE:joselynzhao.top
@SOFTWERE:PyCharm
@FILE:main01.py
@TIME:2019/6/24 09:54
@DES:
'''

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
import tensorflow.contrib.slim as slim

from tensorflow.python.framework import graph_util

from tensorflow.examples.tutorials.mnist import input_data
from sklearn.utils import shuffle

mnist = input_data.read_data_sets('../../../data/mnist', one_hot=True)





if __name__=="__main__":
    data = get_test_data(mnist.test, 9000)
    print(len(data[0]))


