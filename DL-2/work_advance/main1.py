#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn Zhao
@CONTACT:zhaojing17@foxmail.com
@HOME_PAGE:joselynzhao.top
@SOFTWERE:PyCharm
@FILE:main1.py
@TIME:2019/5/6 15:45
@DES:
'''

import matplotlib.pyplot as plt
import numpy as np
import math
import tensorflow as tf
from tensorflow.python.framework import graph_util

if __name__ =="__main__":
    N = 1000
    x = np.linspace(-100,100,N)
    y = 1-np.sin(x)/x

    plt.plot(x,y)
    plt.show()
