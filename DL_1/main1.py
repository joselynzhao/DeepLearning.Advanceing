#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn
@CONTACT:zhaojing17@foxmail.com
@SOFTWERE:PyCharm
@FILE:main1.py
@TIME:2019/4/25 23:06
@DES:
'''

import cv2
from matplotlib import pyplot as plt
img_name = 'dl1.png'
img_name2 = 'dl1-1.png'

import numpy as np
if __name__ == '__main__':

    # 读入图片：默认彩色图，cv2.IMREAD_GRAYSCALE灰度图，cv2.IMREAD_UNCHANGED包含alpha通道

    img = cv2.imread(img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # b, g, r = cv2.split(img)
    # img = cv2.merge([r, g, b])
    plt.title('img_RGB')
    plt.imshow(img)
    plt.show()

    # cv2.imshow('src', img)
    # # cv2.waitKey()
    # k = cv2.waitKey(0) # waitkey代表读取键盘的输入，括号里的数字代表等待多长时间，单位ms。 0代表一直等待
    # if k ==27:     # 键盘上Esc键的键值
    #     cv2.destroyAllWindows()
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    # r, g, b = cv2.split(img)
    img1 = cv2.merge([b,r,g])
    # img1 = np.array(img1)
    plt.title('img_BRG')
    plt.imshow(img1)
    plt.show()

    # test_num = np.array([[[2,3],[3,4]]])
    # print(test_num.shape)
    # print(img.shape)  # (h,w,c)
    school_number = 18023032
    x1=18
    y1=2
    x2=30+18
    y2=32+2

    # l,w,h = img.shape
    # print l,w,h
    img = cv2.imread(img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img[x1:x2+1,y1,1] = (255,0,0)
    img[x1:x2+1,y2,:] = (255,0,0)
    img[x1,y1:y2+1,:] = (255,0,0)
    img[x2,y1:y2+1,:] = (255,0,0)
    cv2.imwrite(img_name2,img)

    img2 = cv2.imread(img_name2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    #
    # b, g, r = cv2.split(img2)
    # img2 = cv2.merge([r, g, b])
    plt.title('img2_RGB')
    plt.imshow(img2)
    plt.show()



    # im_frame[18:49, 2] = (255, 0, 0)
    # im_frame[18:49, 35] = (255, 0, 0)
    # im_frame[18, 2:36] = (255, 0, 0)
    # im_frame[48, 2:36] = (255, 0, 0)
    # cv2.imwrite('im_frame.jpg', im_frame)



    # for i in range(l):
    #     for j in range(w):
    #         if(i in (x1,x2) and j in (y1,y2)):
    #             img[i][j][0]
    # #
    # print(img.size)  # 像素总数目
    #
    # print(img.dtype)
    #
    # print(img)
    #



