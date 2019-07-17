#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn Zhao
@CONTACT:zhaojing17@foxmail.com
@HOME_PAGE:joselynzhao.top
@SOFTWERE:PyCharm
@FILE:main.py
@TIME:2019/7/15 10:33
@DES:
'''
import  tensorflow as tf
import  numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data



def model(x):
    w1 = tf.Variable(dtype=tf.float32, initial_value=np.random.rand(784, 1500),name='w1')
    w2 = tf.Variable(dtype=tf.float32, initial_value=np.random.rand(1500, 1000),name ='w2')
    w3 = tf.Variable(dtype=tf.float32, initial_value=np.random.rand(1000, 500), name='w3')
    w4 = tf.Variable(dtype=tf.float32, initial_value=np.random.rand(500, 10), name='w4')
    b1 = tf.Variable(dtype=tf.float32, initial_value=np.random.rand(1500))
    b2 = tf.Variable(dtype=tf.float32, initial_value=np.random.rand(1000))
    b3 = tf.Variable(dtype=tf.float32, initial_value=np.random.rand(500))
    b4 = tf.Variable(dtype=tf.float32, initial_value=np.random.rand(10))

    fc1 = tf.nn.relu(tf.matmul(x, w1) + b1)
    fc2 = tf.nn.relu(tf.matmul(fc1, w2) + b2)
    fc3 = tf.nn.relu(tf.matmul(fc2, w3) + b3)
    fc4 = tf.matmul(fc3, w4) + b4
    return fc4

def visual_w(sess,w_name):
    w = sess.run(w_name+':0')
    w_min = np.min(w)
    w_max = np.max(w)
    w_0_to_1 = (w - w_min) / (w_max - w_min)
    plt.title(w_name)
    plt.imshow(w_0_to_1)
    plt.show()



if __name__ =="__main__":

    iteratons = 30000
    batch_size = 64
    ma = 0
    sigma = 0.1
    lr = 0.005

    input_image = tf.placeholder(tf.float32, [None, 784])
    input_label = tf.placeholder(tf.float32, [None, 10])

    logits = model(input_image)
    # 注意，使用softmax_cross_entropy_with_logits_v2时，logits对应fc直接输出，不要再加softmax
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=input_label, logits=logits)
    loss = tf.reduce_mean(loss)

    # tv = tf.trainable_variables()
    # lambda_l = 0.0005
    # Regularization_term = lambda_l * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv])
    #
    # loss = Regularization_term + loss

    train_op = tf.train.AdamOptimizer(lr).minimize(loss)

    # 准确率
    correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(logits), 1), tf.argmax(input_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    iteration_index = []
    tr_loss = []
    tr_acc =[]
    valid_acc = []
    valid_loss = []

    with tf.Session() as sess:
        mnist = input_data.read_data_sets('../../../data/mnist', one_hot=True)
        sess.run(tf.global_variables_initializer())
        validation_images = mnist.validation.images
        validation_labels = mnist.validation.labels
        test_images = mnist.test.images
        test_labels = mnist.test.labels
        for ii in range(iteratons):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_op, feed_dict={input_image: batch_xs, input_label: batch_ys})
            if  ii % 100 == 99:
                train_loss,train_accu = sess.run([loss,accuracy], feed_dict={input_image: batch_xs, input_label: batch_ys})
                validation_loss,validation_accu = sess.run([loss,accuracy],
                                           feed_dict={input_image: validation_images, input_label: validation_labels})
                print("Iter [%5d/%5d]: train acc is: %4f train loss is: %4f valid acc is :%4f valid loss is: %4f" % (ii, iteratons, train_accu, train_loss, validation_accu,validation_loss))
                if ii>2000:
                    iteration_index.append(ii)
                    tr_loss.append(train_loss)
                    tr_acc.append(train_accu)
                    valid_acc.append(validation_accu)
                    valid_loss.append(validation_loss)

        acc = sess.run(accuracy,feed_dict={input_image: test_images,input_label: test_labels})
        print("Test: accuracy is %4f" % (acc))

        plt.plot(iteration_index, tr_loss, 'r', label="train_loss")
        plt.plot(iteration_index, valid_loss, 'g',label ="valid_loss")
        plt.xlabel("iteration")
        plt.ylabel("loss")
        plt.legend()
        plt.title("result_loss")
        plt.savefig('./result_loss3.png')
        plt.show()

        plt.plot(iteration_index, tr_acc, 'r', label="train_acc")
        plt.plot(iteration_index, valid_acc, 'g',label ="valid_acc")
        plt.xlabel("iteration")
        plt.ylabel("accuracy")
        plt.legend()
        plt.title("result_acc")
        plt.savefig('./result_acc3.png')
        plt.show()

        plt.figure()
        for i in range(1,5,1):
            w1=sess.run('w'+str(i)+':0')
            w1_min = np.min(w1)
            w1_max = np.max(w1)
            w1_0_to_1 = (w1 - w1_min) / (w1_max - w1_min)
            plt.subplot(2,2,i-1)
            plt.title('w'+str(i))
            plt.imshow(w1_0_to_1)
        plt.savefig('./weight3.png')
        plt.show()