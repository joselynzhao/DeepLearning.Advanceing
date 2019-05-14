#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn
@CONTACT:zhaojing17@foxmail.com
@SOFTWERE:PyCharm
@FILE:main2.py
@TIME:2019/4/27 22:06
@DES:
'''

import cv2
from matplotlib import pyplot as plt
img_name = 'fff.png'
img_name2 = 'fff1.png'

import numpy as np
if __name__ == '__main__':
    img = cv2.imread(img_name)
    cv2.imshow('RGB', img)
    # cv2.waitKey()
    k = cv2.waitKey(0) # waitkey代表读取键盘的输入，括号里的数字代表等待多长时间，单位ms。 0代表一直等待
    if k ==27:     # 键盘上Esc键的键值
        cv2.destroyAllWindows()

    # img = np.array(img)
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]
    # r, g, b = cv2.split(img)
    img = cv2.merge([b, r, g])
    cv2.imshow('BRG', img)
    # cv2.waitKey()
    k = cv2.waitKey(0)  # waitkey代表读取键盘的输入，括号里的数字代表等待多长时间，单位ms。 0代表一直等待
    if k == 27:  # 键盘上Esc键的键值
        cv2.destroyAllWindows()

    school_number = 18023032
    x1 = 18
    y1 = 2
    x2 = 30 + 18
    y2 = 32 + 2
    img[x1:x2 + 1, y1,0] = 0
    img[x1:x2 + 1, y1,1] = 0
    img[x1:x2 + 1, y1,2] = 255
    img[x1:x2 + 1, y2,0] = 0
    img[x1:x2 + 1, y2,1] = 0
    img[x1:x2 + 1, y2,2] = 255
    img[x1, y1:y2 + 1,0] = 0
    img[x1, y1:y2 + 1,1] = 0
    img[x1, y1:y2 + 1,2] = 255
    img[x2, y1:y2 + 1,0] = 0
    img[x2, y1:y2 + 1,1] = 0
    img[x2, y1:y2 + 1,2] = 255

    # img[x1:x2 + 1, y1] = (0, 0, 255)
    # img[x1:x2 + 1, y2] = (0, 0, 255)
    # img[x1, y1:y2 + 1] = (0, 0, 255)
    # img[x2, y1:y2 + 1] = (0, 0, 255)
    cv2.imwrite(img_name2, img)
    img2 = cv2.imread(img_name2)
    cv2.imshow('BRG-1', img2)
    # cv2.waitKey()
    k = cv2.waitKey(0)  # waitkey代表读取键盘的输入，括号里的数字代表等待多长时间，单位ms。 0代表一直等待
    if k == 27:  # 键盘上Esc键的键值
        cv2.destroyAllWindows()

