#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn
@CONTACT:zhaojing17@foxmail.com
@SOFTWERE:PyCharm
@FILE:Scikit-learn1.py
@TIME:2019/5/5 16:35
@DES:使用 Scikit-learn 进行线性回归
'''


if __name__ =="__main__":
    import sklearn
    import tensorflow as tf
    X_train,X_test,y_train,y_test = sklearn.model_selection.train_test_split(X,y,test_size = 0.2)
    # 随机划分20%的数据作为测试集

    clf = sklearn.linear_model.LinearRegression()
    # 定义线性回归器

    clf.fit(X_train,y_train) #开始训练
    accuracy = clf.score(X_test,y_test) #测试并得到测试集性能

