#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn
@CONTACT:zhaojing17@foxmail.com
@SOFTWERE:PyCharm
@FILE:main.py
@TIME:2019/4/24 22:58
@DES:

'''

import cv2
import numpy
from matplotlib import pyplot as plt

# img=cv2.imread("test.jpg")
# img = cv2.resize(img, (100, 100))
# cv2.namedWindow("Image")
# cv2.imshow("Image",img)
# k = cv2.waitKey(0) # waitkey代表读取键盘的输入，括号里的数字代表等待多长时间，单位ms。 0代表一直等待
# if k ==27:     # 键盘上Esc键的键值
#     cv2.destroyAllWindows()

# cv2.waitKey(0)
# cv2.destroyAllWindows()

img=cv2.imread('lena.jpg')
plt.imshow(float(img))
plt.show()