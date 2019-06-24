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

    image1 = []  # 存在第一个图像
    image2 = []  # 存放第二个图像
    label = []  # 存在label
    count_list_right = {}  # 设计为字典
    count_list_nega = {}  # 设计为字典

    '''下面考虑正例，每个数字占（size/2）的1/10'''
    num_each_right = size/2/10  #每个数字创造的样本数
    for i in range(10):  #对每个数字做遍历
        isbreak = 0
        count =0 #当前样本数为0
        len_i = len(from_0_to_9[i]) # 获取当前这个数字对应的样本数量
        for j in  range(len_i): # 对这个数字对应的样本做遍历
            if isbreak:
                break
            for k in range(len_i): # 对这个数字对应的样本进行二重遍历
                image1.append(from_0_to_9[i][j])
                image2.append(from_0_to_9[i][k])
                label.append(0)
                count+=1
                if count>=num_each_right:
                    count_list_right["%d,%d"%(i,i)]=count
                    isbreak = 1
                    break
                    #跳出两层for循环
    print(count_list_right)
    print(len(count_list_right))
    # {'9,9': 5, '7,7': 5, '4,4': 5, '0,0': 5, '3,3': 5, '8,8': 5, '5,5': 5, '1,1': 5, '6,6': 5, '2,2': 5}
    '''下面考虑负样本'''
    num_each_nega = size/2/45
    for i in range(10): # 对0-9遍历
        for j in range(i+1,10): #仍对0-9遍历
            # if j==i:
            #     continue #跳过两个数字相同的情况
            # 下面对于两个不同的数字来做考虑
            # len_i = len(from_0_to_9[i])
            # len_j = len(from_0_to_9[j])  # 分别获取两个数字的样本个数  。可能会用不着
            for count in range(num_each_nega): #构造这num_each_nega多个样本数据
                image1.append(from_0_to_9[i][count])
                image2.append(from_0_to_9[j][count])
                label.append(1)
            count_list_nega["%d,%d"%(i,j)]=num_each_nega

    print(count_list_nega)
    print(len(count_list_nega))

    '''组合数据'''
    data = []
    data.append(image1)
    data.append(image2)
    data.append(label)
    return data



if __name__=="__main__":
    data = get_test_data(mnist.test, 9000)
    print(len(data[0]))


