#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn Zhao
@CONTACT:zhaojing17@foxmail.com
@HOME_PAGE:joselynzhao.top
@SOFTWERE:PyCharm
@FILE:example_vgg16.py
@TIME:2019/5/16 16:30
@DES:
'''

 #vgg16.py
class vgg16(object):
    def __init__(self):
        #当在创建的时候运行画图
        self._build_graph()

    def fix_conv1_from_RGB_to_BGR(self,sess):
        #外 部 调 用 的 计 算 过 程， 需 要 传 递 相 关 的sess会 话
        restorer_conv1.restore(sess, pretrained_model)
        sess.run(tf.assign(var_to_fix[...],new_value))


#main.py

net=vgg16() #此时整个计算图里已经添加了vgg16_obj的计算图 ....
sess.run(init) #先 初 始 化 好 变 量
#再恢复一些需要的预训练值
restorer.restore(sess, pretrained_model)
#再修改一些值
net.fix_conv1_from_RGB_to_BGR(sess, pretrained_model)