#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn Zhao
@CONTACT:zhaojing17@foxmail.com
@HOME_PAGE:joselynzhao.top
@SOFTWERE:PyCharm
@FILE:main.py
@TIME:2019/5/14 10:44
@DES:
'''


import  tensorflow as tf
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
from matplotlib import pyplot as plt
from scipy import misc
import pylab
import scipy
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow.python.framework import graph_util


from tensorflow.examples.tutorials.mnist import input_data


# side_width = 28
# size = side_width*side_width/2

if __name__ =="__main__":
    input = mpimg.imread('test.png')  # 读取和代码处于同一目录下的 lena.pn
    image = np.array(ndimage.imread("test.png", flatten=False))
    print(image.shape)

    my_image = scipy.misc.imresize(image, size=(28,28))
    x1 = my_image[0:14,]
    x2 = my_image[14:28,]

    # 可以成功的将图片分割为两部分
    # print(my_image.shape)
    # print(x1.shape)
    # print(x2.shape)
    # plt.imshow(my_image)
    # pylab.show()
    # plt.imshow(x1)
    # pylab.show()
    # plt.imshow(x2)
    # pylab.show()

    # plt.imshow(input)  # 显示图片
    # plt.axis('off')  # 不显示坐标轴
    # plt.show()




    input1 = tf.placeholder(tf.float32,[None,size])
    input2 = tf.placeholder(tf.float32,[None,size])
    y_ = tf.placeholder(dtype='float',shape=[None,10])

    def X_W(x):
        W = tf.get_variable(tf.zeros([size,10]),name='w')
        y = tf.matmul(x,W)
        return y

    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(X_W(x1)+X_W(x2)+b)

    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cross_entropy)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)







