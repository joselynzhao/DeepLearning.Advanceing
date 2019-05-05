#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn
@CONTACT:zhaojing17@foxmail.com
@SOFTWERE:PyCharm
@FILE:PB_save1.py
@TIME:2019/5/5 18:04
@DES:
'''
save_file ="./save/pbplus.pb"

if __name__ =="__main__":
    import tensorflow as tf
    from tensorflow.python.framework import graph_util
    x = tf.Variable(tf.random_uniform([3]))
    y = tf.Variable(tf.random_uniform([3]))
    z = tf.add(x, y, name='op_to_store')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(x))
        print(sess.run(y))
        print(sess.run(z))
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['op_to_store'])
        with tf.gfile.FastGFile(save_file, mode='wb') as f:
            f.write(constant_graph.SerializeToString())

    '''[0.5625318  0.71519125 0.34229362]
    [0.49225044 0.16457498 0.53800344]
    [1.0547823  0.8797662  0.88029706]
    Converted 2 variables to const ops.'''


    # pb 恢复
    import tensorflow as tf
    from tensorflow.python.platform import gfile
    # ...... something disappeared ......

    with tf.Session() as sess:
        with gfile.FastGFile(save_file, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')
        sess.run(tf.global_variables_initializer())
        z = sess.graph.get_tensor_by_name('op_to_store:0') # x? y?
        print(sess.run(z))

    '''[1.0547823  0.8797662  0.88029706]'''
    '''只取出了z的值'''