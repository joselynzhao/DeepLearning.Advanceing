#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn Zhao
@CONTACT:zhaojing17@foxmail.com
@HOME_PAGE:joselynzhao.top
@SOFTWERE:PyCharm
@FILE:example_jicheng.py
@TIME:2019/5/16 16:55
@DES:
'''

#network.py
class Network(object):
    def __init__(self):
        self._layers={}

    def _img_to_classifier1(self,is_training, reuse=None):
        raise NotImplementedError
    def _add_train_summary(self, var):
        #子 类 共 性 操 作， 父 类 已 具 体 实 现
        pass
    def _add_losses(self, var):
        #子 类 共 性 操 作， 父 类 已 具 体 实 现
        pass

    #vgg16.py
class vgg16(Network):
    # Network 父类中的其他方法,例如:_add_train_summary,_add_losses等vgg16都可以调用,vgg16类只用写其相比
    #Network不同的方法

    def __init__(self):

        Network.__init__(self)  # 初 始 化 时， 先 调 用 父 类 初 始 化 方 法

    def _img_to_classifier1(self, is_training, reuse=None):
        # 具体实现覆盖父类方法
        pass
