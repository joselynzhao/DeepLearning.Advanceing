#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn
@CONTACT:zhaojing17@foxmail.com
@SOFTWERE:PyCharm
@FILE:Session1.py
@TIME:2019/5/5 15:55
@DES: 会话session 练习
'''



if __name__ == "__main__":
    import tensorflow as tf
    v1 = tf.constant([1.0,2.0,3.0],shape=[3],name='v1')
    v2 = tf.constant([1.0,2.0,3.0],shape=[3],name='v2')
    sum12 = v1+v2

    with tf.Session(config=tf.ConfigProto(log_device_placement = True)) as sess:
        print sess.run(sum12)

    '''ConfigProto(log_device_placement = True) 的目的是为了在输出中指明cpu'''
    '''运行结果如下：
    add: (Add): /job:localhost/replica:0/task:0/device:CPU:0
    v2: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    v1: (Const): /job:localhost/replica:0/task:0/device:CPU:0
    [2. 4. 6.]
    '''


    #   手动指定调用某个CPU或者GPU
    import tensorflow as tf

    with tf.device('/cpu:0'):
        v1 = tf.constant([1.0, 2.0, 3.0], shape=[3], name='v1')
        v2 = tf.constant([1.0, 2.0, 3.0], shape=[3], name='v2')
        sum12 = v1 + v2

        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            print sess.run(sum12)

    #会话模式1
    sess = tf.Session()
    sess.run()
    sess.close()
    #会话模式2
    with tf.Session() as sess:
        sess.run()


