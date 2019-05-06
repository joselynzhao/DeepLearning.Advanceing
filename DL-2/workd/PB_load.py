#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn
@CONTACT:zhaojing17@foxmail.com
@SOFTWERE:PyCharm
@FILE:PB_load.py
@TIME:2019/5/6 10:00
@DES: PB模型恢复程序。
'''

import  tensorflow as tf
import  numpy as np
import  math
import matplotlib.pyplot as plt
from tensorflow.python.platform import gfile

save_path_pb = './save/dl-2-work.pb'


if __name__ =="__main__":
    with tf.Session() as sess:
        with gfile.FastGFile(save_path_pb, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')
        sess.run(tf.global_variables_initializer())
        w1 = sess.graph.get_tensor_by_name('s_w1:0')
        w2 = sess.graph.get_tensor_by_name('s_w2:0')
        w3 = sess.graph.get_tensor_by_name('s_w3:0')
        b = sess.graph.get_tensor_by_name('s_b:0')
        print(sess.run(w1))
        print(sess.run(w2))
        print(sess.run(w3))
        print(sess.run(b))
        w1 = w1.eval()
        w2 = w2.eval()
        w3 = w3.eval()
        b = b.eval()

    school_number = 18023032
    aa = 18.0
    bb = 32.0
    N = 2000
    x1 = np.linspace(-bb / aa, (2 * math.pi - bb) / aa, N)
    y1 = np.cos(aa * x1 + bb)
    y2 = x1 * w1 + (x1 ** 2) * w2 + (x1 ** 3) * w3 + b
    y2 = np.reshape(y2, [-1, 1])
    print(x1.shape)
    print(y1.shape)
    print(y2.shape)
    plt.plot(x1, y1, 'r')
    plt.plot(x1, y2, 'g')
    plt.title("test")
    plt.show()