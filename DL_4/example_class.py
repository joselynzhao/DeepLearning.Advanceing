#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn Zhao
@CONTACT:zhaojing17@foxmail.com
@HOME_PAGE:joselynzhao.top
@SOFTWERE:PyCharm
@FILE:example_class.py
@TIME:2019/5/16 15:35
@DES:
'''

#创建类
class Gre:
#类中的方法
    def Test(self): #类中方法的第一个参数必须是self，代表类的实例,
    #类似于C++中的this指针(self也可以统一换成其他名字)!
        pass
        #空 语 句， 啥 也 不 干， 保 持 程 序 结 构 完 整 性
        #定 义 空 函 数 会 报 错， 没 想 好 写 啥 就 先pass
    def Hi(self):
        print('Hi')
    def Hello(self,name):
        print("I'm␣%s" %name)

#根据类Gre声明 创建或说实例化类Gre的一个对象obj
obj=Gre() #实 例 化 一 定 要 加'()'!
obj.Hi() #调 用Hi方 法
obj.Hello('Han␣Meimei') #调 用Hello方 法