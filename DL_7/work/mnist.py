#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn Zhao
@CONTACT:zhaojing17@foxmail.com
@HOME_PAGE:joselynzhao.top
@SOFTWERE:PyCharm
@FILE:mnist.py
@TIME:2019/6/24 19:53
@DES:
'''

import  tensorflow as tf

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

def pre_data(images,labels,size):
    '''
    :param data: 待处理的数据集
    :param size: 目标样本数量
    :return: 处理后的数据集
    '''

    '''分别取出10是数字'''
    len_data = len(images)
    from_0_to_9 = [[],[],[],[],[],[],[],[],[],[]]
    for i in range(len_data): #对数据集进行遍历
        for j in range(10): #对0-9进行遍历
            if labels[i][j] == 1:
                from_0_to_9[j].append(images[i])

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
                if j==k :
                    continue
                image1.append(from_0_to_9[i][j])
                image2.append(from_0_to_9[i][k])
                label.append([0])
                count+=1
                if count>=num_each_right:
                    count_list_right["%d,%d"%(i,i)]=count
                    isbreak = 1
                    break
                    #跳出两层for循环
    # print(count_list_right)
    # print(len(count_list_right))
    # {'9,9': 5, '7,7': 5, '4,4': 5, '0,0': 5, '3,3': 5, '8,8': 5, '5,5': 5, '1,1': 5, '6,6': 5, '2,2': 5}
    '''下面考虑负样本'''
    num_each_nega = size/2/45
    for i in range(10): # 对0-9遍历
        for j in range(i+1,10): #仍对0-9遍历
            for count in range(num_each_nega): #构造这num_each_nega多个样本数据
                image1.append(from_0_to_9[i][count])
                image2.append(from_0_to_9[j][count])
                label.append([1])
            count_list_nega["%d,%d"%(i,j)]=num_each_nega

    # print(count_list_nega)
    # print(len(count_list_nega))

    '''组合数据'''
    data = []
    data.append(image1)
    data.append(image2)
    data.append(label)
    return data


''' 下面设计网络部分'''

def fully_connected_layer(scope_name,x,W_name,b_name,W_shape,reuse = False):
    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()
        fc_W = tf.get_variable(W_name, initializer=tf.truncated_normal(W_shape, stddev=0.1))
        fc_b = tf.get_variable(b_name, initializer=tf.zeros(W_shape[1]))
        fc = tf.matmul(x, fc_W) + fc_b
        return fc

def net(x,scope_name='net'):
    with tf.variable_scope(scope_name,reuse=tf.AUTO_REUSE):
        fc0 = fully_connected_layer("fc1",x,"fc1_w","fc1_b",[784,500])
        fc0 = tf.nn.relu(fc0)
        fc1 = fully_connected_layer("fc2",fc0,"fc2_w","fc2_b",[500,10])
        fc1 = tf.nn.relu(fc1)
        return fc1

def model():
    Q = tf.constant([5.0])
    thresh = 1.50  # 用于判断的距离阈值
    iterations = 20000
    lr = 0.1
    batch_size = 900

    x1 = tf.placeholder(tf.float32, [None, 784], name="x1")
    x2 = tf.placeholder(tf.float32, [None, 784], name="x2")
    y = tf.placeholder(tf.float32, [None,1], name="y")
    # y2 = tf.placeholder(tf.float32, [None, 10], name="y2")
    net1 = net(x1)
    net2 = net(x2)
    Ew = tf.sqrt(tf.reduce_sum(tf.square(net1 - net2), 1))  #我可以把它理解为计算结果么？
    L1 = 2 * (1 - y) * tf.square(Ew) / Q
    L2 = 2 * y * tf.exp(-2.77 * Ew / Q) * Q
    Loss = tf.reduce_mean(L1 + L2)

    prediction = tf.greater(Ew, thresh)  #这个函数又是个什么鬼啊
    # 定义准确率
    correct_prediction = tf.equal(prediction, tf.cast(y, tf.bool))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 采用Adam作为优化器
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(Loss)
    # # 定义保存器
    # saver = tf.train.Saver(max_to_keep=4)
    test_images = mnist.test.images
    test_labels = mnist.test.labels
    test_data = pre_data(test_images,test_labels,9000)

    image_x = []
    image_y_acc = []
    image_y_loss = []

    with tf.Session() as sess:  #这个session需要关闭么？
        sess.run(tf.global_variables_initializer())
        for i in range(iterations):
            images,labels = mnist.train.next_batch(batch_size)
            data = pre_data(images,labels,batch_size)
            sess.run(train_step,feed_dict={x1:data[0],x2:data[1],y:data[2]})
            if i%10 == 1:
                loss,acc = sess.run([Loss,accuracy],feed_dict={x1:test_data[0],x2:test_data[1],y:test_data[2]})
                print("%5d: accuracy is: %4f , loss is : %4f 。" % (i, acc, loss))
                image_x.append(i)
                image_y_acc.append(acc)
                image_y_loss.append(loss)
        plt.plot(image_x, image_y_acc, 'r', label="accuracy")
        plt.plot(image_x, image_y_loss, 'g', label="loss")
        plt.xlabel("iteration")
        plt.ylabel("accuracy")
        plt.title("acc_loss_v2")
        plt.savefig('./save/acc_loss_v2.png')
        plt.show()
        print( '[accuracy,loss]:', sess.run([accuracy,Loss],feed_dict={x1:test_data[0],x2:test_data[1],y:test_data[2]}))


if __name__ =="__main__":
    model()