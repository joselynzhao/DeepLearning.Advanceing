#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn
@CONTACT:zhaojing17@foxmail.com
@SOFTWERE:PyCharm
@FILE:ckpt_save1.py
@TIME:2019/5/5 17:50
@DES:
'''

save_file = './save/ckpt1.ckpt'

if __name__ =="__main__":
    import tensorflow as tf

    x = tf.Variable(tf.random_uniform([3]))
    y = tf.Variable(tf.random_uniform([3]))
    z = tf.add(x, y)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(x))
        print(sess.run(y))
        print(sess.run(z))
        save_path = saver.save(sess,save_file)

    '''运行结果：
    [0.6390506  0.26704168 0.09797013]
    [0.98880136 0.55906487 0.00470507]
    [1.627852   0.82610655 0.1026752 ]
    并在save目录下参数相应的文件'''


    # 模型恢复
    import tensorflow as tf

    x = tf.Variable(tf.random_uniform([3]))
    y = tf.Variable(tf.random_uniform([3]))
    z = tf.add(x, y)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, save_file)
        print(sess.run(x))
        print(sess.run(y))
        print(sess.run(z))

    '''运行结果：
    [0.6390506  0.26704168 0.09797013]
    [0.98880136 0.55906487 0.00470507]
    [1.627852   0.82610655 0.1026752 ]
    与刚才存储的结果完全一样'''