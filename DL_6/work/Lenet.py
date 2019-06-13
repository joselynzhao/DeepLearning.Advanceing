#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn Zhao
@CONTACT:zhaojing17@foxmail.com
@HOME_PAGE:joselynzhao.top
@SOFTWERE:PyCharm
@FILE:Lenet.py
@TIME:2019/6/11 18:20
@DES:
'''


import  tensorflow as tf


def conv_image_visual(conv_image, image_weight, image_height, cy, cx, channels):
    # slice off one image ande remove the image dimension
    # original image is a 4d tensor[batche_size,weight,height,channels]
    conv_image = tf.slice(conv_image, (0, 0, 0, 0), (1, -1, -1, -1))
    conv_image = tf.reshape(conv_image, (image_height, image_weight, channels))
    # add a couple of pixels of zero padding around the image
    image_weight += 4
    image_height += 4
    conv_image = tf.image.resize_image_with_crop_or_pad(conv_image, image_height, image_weight)
    conv_image = tf.reshape(conv_image, (image_height, image_weight, cy, cx))
    conv_image = tf.transpose(conv_image, (2, 0, 3, 1))
    conv_image = tf.reshape(conv_image, (1, cy * image_height, cx * image_weight, 1))
    return conv_image


def variable_summaries(var,name):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/'+name,mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev/'+name,stddev)
        tf.summary.histogram(name,var)



class Lenet():
    def __init__(self,mu,sigma,lr=0.02,act = 'relu'):
        self.mu = mu
        self.sigma = sigma
        self.lr = lr
        self.activation = act  # 默认是relu
        self._build_graph()

    def change_act(self,act):
        self.activation = act


    def _build_graph(self,network_name = "Lenet"):
        self._setup_placeholders_graph()
        self._build_network_graph(network_name)
        self._compute_loss_graph()
        self._compute_acc_graph()
        self._create_train_op_graph()

    def _setup_placeholders_graph(self):
        self.x  = tf.placeholder("float",shape=[None,32,32,1],name='x')
        self.y_ = tf.placeholder("float",shape = [None,10],name ="y_")

    def _cnn_layer(self,scope_name,W_name,b_name,x,filter_shape,conv_stride,padding_tag="VALID",reuse=False):
        with tf.variable_scope(scope_name) as scope:
            if reuse:
                scope.reuse_variables()
            conv_W = tf.Variable(tf.truncated_normal(shape=filter_shape, mean=self.mu, stddev=self.sigma), name=W_name)
            # self.variable_summaries(conv_W)  #可视化
            conv_b = tf.Variable(tf.zeros(filter_shape[3]),name=b_name)
            # self.variable_summaries(conv_b)  #可视化
            # conv_b = tf.Variable(tf.constant(0.1,shape=filter_shape[3]),name=b_name)
            conv = tf.nn.conv2d(x, conv_W, strides=conv_stride, padding=padding_tag) + conv_b
            tf.summary.histogram("weights",conv_W)
            tf.summary.histogram("biases",conv_b)
            self._conv_visual(conv,conv_W,filter_shape)  #可视化
            return conv

    def _pooling_layer(self,scope_name,x,pool_ksize,pool_strides,padding_tag="VALID",reuse=False):
        with tf.variable_scope(scope_name) as scope:
            if reuse:
                scope.reuse_variables()
            pool = tf.nn.max_pool(x, ksize=pool_ksize, strides=pool_strides, padding=padding_tag)
            return pool

    def _fully_connected_layer(self,scope_name,W_name,b_name,x,W_shape,reuse=False):
        with tf.variable_scope(scope_name) as scope:
            if reuse:
                scope.reuse_variables()
            fc_W = tf.Variable(tf.truncated_normal(shape=W_shape, mean=self.mu, stddev=self.sigma),name=W_name)
            fc_b = tf.Variable(tf.zeros(W_shape[1]),name=b_name)
            fc = tf.matmul(x, fc_W) + fc_b
            tf.summary.histogram("weights",fc_W)
            tf.summary.histogram("biases",fc_b)
            self._full_visual(fc_W,W_shape)  #可视化

            return fc

    def _build_network_graph(self,scope_name):
        with tf.variable_scope(scope_name):
            conv1 =self._cnn_layer("conv1","w1","b1",self.x,[5,5,1,6],[1, 1, 1, 1])
            self.conv1 = self._activation_way(conv1)
            self.pool1 = self._pooling_layer("pool1",self.conv1,[1, 2, 2, 1],[1, 2, 2, 1])

            conv2 = self._cnn_layer("conv2","w2","b2",self.pool1,[5,5,6,16],[1, 1, 1, 1])
            self.conv2 = self._activation_way(conv2)
            self.pool2 = self._pooling_layer("pool2",self.conv2,[1, 2, 2, 1],[1, 2, 2, 1])

            self.fc0 = self._flatten(self.pool2)

            fc1 = self._fully_connected_layer("fc1","wfc1","bfc1",self.fc0,[400,120])
            self.fc1 = self._activation_way(fc1)

            fc2 = self._fully_connected_layer("fc2","wfc2","bfc2",self.fc1,[120,84])
            self.fc2 = self._activation_way(fc2)

            self.y = self._fully_connected_layer("fc3","wfc3","bfc3",self.fc2,[84,10])
            tf.summary.histogram("ypredict",self.y)

    def _activation_way(self,layer):
        if (self.activation == "relu"):
            layer = tf.nn.relu(layer)
        elif (self.activation =="sigmoid"):
            layer = tf.nn.sigmoid(layer)
        return layer #返回激活后的层


    def _flatten(self,conv):
        conv1 = tf.reshape(conv, [-1, 400])
        return conv1

    def _compute_loss_graph(self):
        with tf.name_scope("loss_function"):
            loss = tf.nn.softmax_cross_entropy_with_logits(labels = self.y_,logits = self.y)
            self.loss = tf.reduce_mean(loss)
            tf.summary.scalar("loss",self.loss)

    def _compute_acc_graph(self):
        with tf.name_scope("acc_function"):
            correct_prediction = tf.equal(tf.argmax(self.y,1),tf.argmax(self.y_,1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
            tf.summary.scalar("accuracy", self.accuracy)

    def _create_train_op_graph(self):
        with tf.name_scope("train_function"):
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y,labels=self.y_))
            self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.cross_entropy)
            tf.summary.scalar("cross_entropy",self.cross_entropy)

    def _conv_visual(self,conv,conv_W, filter_shape):
        with tf.name_scope('visual'):
            # scale weights to [0 1], type is still float
            x_min = tf.reduce_min(conv_W)
            x_max = tf.reduce_max(conv_W)
            kernel_0_to_1 = (conv_W - x_min) / (x_max - x_min)
            # to tf.image_summary format [batch_size, height, width, channels]
            # this will display random 3 filters from the 64 in conv1
            kernel_transposed = tf.transpose(kernel_0_to_1, [3, 2, 0, 1])
            conv_W_img = tf.reshape(kernel_transposed, [-1, filter_shape[0], filter_shape[1], 1])
            tf.summary.image('conv_w', conv_W_img, max_outputs=filter_shape[3])
            feature_img = conv[0:1, :, :, 0:filter_shape[3]]
            feature_img = tf.transpose(feature_img, perm=[3, 1, 2, 0])
            tf.summary.image('feature_conv', feature_img, max_outputs=filter_shape[3])

    def _full_visual(self,fc_W, W_shape):
        with tf.name_scope('visual'):
            x_min = tf.reduce_min(fc_W)
            x_max = tf.reduce_max(fc_W)
            kernel_0_to_1 = (fc_W - x_min) / (x_max - x_min)
            fc_W_img = tf.reshape(kernel_0_to_1, [-1, W_shape[0], W_shape[1], 1])
            tf.summary.image('fc_w', fc_W_img, max_outputs=1)




