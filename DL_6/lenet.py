#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn Zhao
@CONTACT:zhaojing17@foxmail.com
@HOME_PAGE:joselynzhao.top
@SOFTWERE:PyCharm
@FILE:lenet.py
@TIME:2019/6/1 16:14
@DES:
'''


import  tensorflow as tf

class Lenet():
    def __init__(self,mu,sigma,lr=0.02):
        self.mu = mu
        self.sigma = sigma
        self.lr = lr
        self._build_graph()


    def _build_graph(self,network_name = "Lenet"):
        self._setup_placeholders_graph()
        self._build_network_graph(network_name)
        self._compute_loss_graph()
        self._compute_acc_graph()
        self._create_train_op_graph()

    def _setup_placeholders_graph(self):
        self.x  = tf.placeholder("float",shape=[None,32,32,1],name='x')
        self.y_ = tf.placeholder("float",shape = [None,10],name ="y_")

    def _cnn_layer(self,scope_name,W_name,b_name,x,filter_shape,conv_stride,padding_tag="VALID"):
        with tf.variable_scope(scope_name):
            conv_W = tf.Variable(tf.truncated_normal(shape=filter_shape, mean=self.mu, stddev=self.sigma), name=W_name)
            conv_b = tf.Variable(tf.zeros(filter_shape[3]),name=b_name)
            conv = tf.nn.conv2d(x, conv_W, strides=conv_stride, padding=padding_tag) + conv_b
            tf.summary.histogram("weights",conv_W)
            tf.summary.histogram("biases",conv_b)

            return conv

    def _pooling_layer(self,scope_name,x,pool_ksize,pool_strides,padding_tag="VALID"):
        with tf.variable_scope(scope_name):
            pool = tf.nn.max_pool(x, ksize=pool_ksize, strides=pool_strides, padding=padding_tag)
            return pool
    def _fully_connected_layer(self,scope_name,W_name,b_name,x,W_shape):
        with tf.variable_scope(scope_name):
            fc_W = tf.Variable(tf.truncated_normal(shape=W_shape, mean=self.mu, stddev=self.sigma),name=W_name)
            fc_b = tf.Variable(tf.zeros(W_shape[1]),name=b_name)
            fc = tf.matmul(x, fc_W) + fc_b
            tf.summary.histogram("weights",fc_W)
            tf.summary.histogram("biases",fc_b)
            return fc

    def _build_network_graph(self,scope_name):
        with tf.variable_scope(scope_name):
            conv1 =self._cnn_layer("conv1","w1","b1",self.x,[5,5,1,6],[1, 1, 1, 1])
            act1 = tf.nn.relu(conv1)
            tf.summary.histogram("activations", act1)
            self.conv1 = act1
            self.pool1 = self._pooling_layer("pool1",self.conv1,[1, 2, 2, 1],[1, 2, 2, 1])
            conv2 = self._cnn_layer("conv2","w2","b2",self.pool1,[5,5,6,16],[1, 1, 1, 1])
            act2 = tf.nn.relu(conv2)
            self.conv2 = act2
            tf.summary.histogram("activations", act2)
            self.pool2 = self._pooling_layer("pool2",self.conv2,[1, 2, 2, 1],[1, 2, 2, 1])
            self.fc0 = self._flatten(self.pool2)
            fc1 = self._fully_connected_layer("fc1","wfc1","bfc1",self.fc0,[400,120])
            act3 = tf.nn.relu(fc1)
            tf.summary.histogram("activations",act3)
            self.fc1 = act3
            fc2 = self._fully_connected_layer("fc2","wfc2","bfc2",self.fc1,[120,84])
            act4 = tf.nn.relu(fc2)
            tf.summary.histogram("activations",act4)
            self.fc2 = act4
            self.y = self._fully_connected_layer("fc3","wfc3","bfc3",self.fc2,[84,10])

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












