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


def get_test_data(data,size):
    '''
    :param data: 待处理的数据集
    :param size: 目标样本数量
    :return: 处理后的数据集
    '''

    '''分别取出10是数字'''
    len_data = len(data.images)
    from_0_to_9 = [[],[],[],[],[],[],[],[],[],[]]
    for i in range(len_data): #对数据集进行遍历
        for j in range(10): #对0-9进行遍历
            if data.labels[i][j] == 1:
                from_0_to_9[j].append(data.images[i])

    # for i in range(10):
    #     print(len(from_0_to_9[i]))
    '''980
    1135
    1032
    1010
    982
    892
    958
    1028
    974
    1009'''

    '''下面考虑正例，每个数字占（size/2）的1/10'''
    num_each = size/2/10  #每个数字创造的样本数
    image1=[] #存在第一个图像
    image2=[]  #存放第二个图像
    label = [] #存在label
    count_list={} #设计为字典
    for i in range(10):  #对每个数字做遍历
        count =0 #当前样本数为0
        len_i = len(from_0_to_9[i]) # 获取当前这个数字对应的样本数量
        for j in  range(len_i): # 对这个数字对应的样本做遍历
            for k in range(len_i): # 对这个数字对应的样本进行二重遍历
                image1.append(from_0_to_9[j])
                image2.append(from_0_to_9[k])
                label.append(0)
                count+=1
                if count>=num_each:
                    count_list["i"+'i']=count
                    #跳出两层for循环
            



if __name__=="__main__":
    get_test_data(mnist.test, 100)

