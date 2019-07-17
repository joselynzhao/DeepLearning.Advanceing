#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn Zhao
@CONTACT:zhaojing17@foxmail.com
@HOME_PAGE:joselynzhao.top
@SOFTWERE:PyCharm
@FILE:Script.py
@TIME:2019/6/29 21:58
@DES:
'''


from mnist2 import *

if __name__ =="__main__":
    size = 9000
    test_images = mnist.test.images
    test_labels = mnist.test.labels
    test_data,count_right,count_nega = pre_data(test_images, test_labels, size)
    num_each_right = int(size / 2 / 10)  # 每个数字创造的样本数
    num_each_nega = int(size / 2 / 45)
    print "正例：",count_right
    print "正例的总样本数为",len(count_right)*num_each_right
    print "反例：",count_nega
    print "正例的总样本数为", len(count_nega) * num_each_nega
    print "测试集长度为：", len(test_data[0])
