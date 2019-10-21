# -*- coding:utf-8 _*-   
# author: tangwei 
# time: 2018/09/20 11：20

'''
    求模型在训练集和测试集的准确率
'''

#coding=utf-8

import tensorflow as tf
import numpy as np
from datetime import datetime
import ihnet_v2
import create_tf_record
import tensorflow.contrib.slim as slim


labels_nums = 6   # 类别个数 6
batch_size = 16  #
resize_height = 512  # 指定存储图片高度
resize_width = 512  # 指定存储图片宽度
depths = 3
data_shape = [batch_size, resize_height, resize_width, depths]

# 定义input_images为图片数据
input_images = tf.placeholder(dtype=tf.float32, shape=[None, resize_height, resize_width, depths], name='input')
# 定义input_labels为labels数据
input_labels = tf.placeholder(dtype=tf.int32, shape=[None, labels_nums], name='label')

# 定义dropout的概率
keep_prob = tf.placeholder(tf.float32,name='keep_prob')
is_training = tf.placeholder(tf.bool, name='is_training')


def net_evaluation(sess, loss, accuracy, val_images_batch, val_labels_batch, val_nums):
    val_max_steps = int(val_nums / batch_size)
    val_losses = []
    val_accs = []
    for _ in range(val_max_steps):
        val_x, val_y = sess.run([val_images_batch, val_labels_batch])
        val_loss,val_acc = sess.run([loss,accuracy], feed_dict={input_images: val_x,
                                                                input_labels: val_y,
                                                                keep_prob: 1.0,
                                                                is_training: False})
        val_losses.append(val_loss)
        val_accs.append(val_acc)
    mean_loss = np.array(val_losses, dtype=np.float32).mean()
    mean_acc = np.array(val_accs, dtype=np.float32).mean()
    return mean_loss, mean_acc


def step_accuracy(loss, accuracy, train_images_batch, train_labels_batch,
                  train_sexs_batch, train_nums, val_images_batch,
                  val_labels_batch, val_sexs_batch, val_nums):
    '''
    循环迭代训练过程
    :param loss:     loss函数
    :param accuracy: 准确率函数
    :param train_images_batch: 训练images数据
    :param train_labels_batch: 训练labels数据
    :param train_nums:         总训练数据
    :param val_images_batch: 验证images数据
    :param val_labels_batch: 验证labels数据
    :param val_nums:         总验证数据
    :return: None
    '''
    # 只加载数据，不加载图结构，可以在新图中改变batch_size等的值
    # 不过需要注意，Saver对象实例化之前需要定义好新的图结构，否则会报错
    saver = tf.train.Saver()

    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.8
    # with tf.Session(config=config) as sess:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # 加载保存模型代码
        ckpt = tf.train.get_checkpoint_state('./logs/')
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            # saver.restore(sess, 'good_model_RSNA/Tg/model.ckpt-11600')

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        train_max_steps = int(train_nums / batch_size)
        train_losses = []
        train_accs = []
        for i in range(train_max_steps):
            batch_input_images, batch_input_labels = sess.run([train_images_batch, train_labels_batch])
            train_acc, train_loss = sess.run([accuracy, loss], feed_dict={input_images: batch_input_images,
                                                                          input_labels: batch_input_labels,
                                                                          keep_prob: 1, is_training: False})
            train_losses.append(train_loss)
            train_accs.append(train_acc)

        train_loss = np.array(train_losses, dtype=np.float32).mean()
        train_acc = np.array(train_accs, dtype=np.float32).mean()
        print("%s: total  train Loss : %f, training accuracy :  %g" % (datetime.now(), train_loss, train_acc))

        # val测试(测试全部val数据)
        mean_loss, mean_acc = net_evaluation(sess, loss, accuracy, val_images_batch, val_labels_batch, val_nums)
        print("%s: total  val Loss : %f, val accuracy :  %g" % (datetime.now(), mean_loss, mean_acc))

        coord.request_stop()
        coord.join(threads)


def accuracy(train_record_file, val_record_file, labels_nums, data_shape):
    '''
    :param train_record_file: 训练的tfrecord文件
    :param train_log_step: 显示训练过程log信息间隔
    :param train_param: train参数
    :param val_record_file: 验证的tfrecord文件
    :param val_log_step: 显示验证过程log信息间隔
    :param val_param: val参数
    :param labels_nums: labels数
    :param data_shape: 输入数据shape
    :param snapshot: 保存模型间隔
    :param snapshot_prefix: 保存模型文件的前缀名
    :return:
    '''
    [batch_size, resize_height, resize_width, depths] = data_shape

    # 获得训练和测试的样本数
    train_nums = create_tf_record.get_txt_nums(train_record_file)
    val_nums = create_tf_record.get_txt_nums(val_record_file)
    print('train nums:%d,val nums:%d'%(train_nums, val_nums))

    # 从record中读取图片和labels数据,sexs数据
    # train数据,训练数据一般要求打乱顺序shuffle=True
    train_images, train_labels = create_tf_record.read_records(train_record_file,
                                                               resize_height,
                                                               resize_width,
                                                               type='normalization')
    train_images_batch, train_labels_batch = create_tf_record.get_batch_images(train_images,
                                                                               train_labels,
                                                                               batch_size=batch_size,
                                                                               shuffle=False)
    # val数据,验证数据可以不需要打乱数据
    val_images, val_labels = create_tf_record.read_records(val_record_file,
                                                           resize_height,
                                                           resize_width,
                                                           type='normalization')
    val_images_batch, val_labels_batch = create_tf_record.get_batch_images(val_images,
                                                                           val_labels,
                                                                           batch_size=batch_size,
                                                                           shuffle=False)

    # Define the model:
    with slim.arg_scope(ihnet_v2.ihnet_arg_scope()):
        out, end_points = ihnet_v2.ihnet(inputs=input_images,
                                         num_classes=labels_nums,
                                         dropout_keep_prob=keep_prob,
                                         is_training=is_training)

    # Specify the loss function: tf.losses定义的loss函数都会自动添加到loss函数,不需要add_loss()了
    tf.losses.softmax_cross_entropy(onehot_labels=input_labels, logits=out)#添加交叉熵损失loss=1.6

    loss = tf.losses.get_total_loss(add_regularization_losses=True)#添加正则化损失loss=2.2
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(input_labels, 1)), tf.float32))

    # 循环迭代过程
    step_accuracy(loss, accuracy, train_images_batch, train_labels_batch,
                  train_nums, val_images_batch, val_labels_batch, val_nums
                  )


if __name__ == '__main__':
    train_record_file = 'train_jpg.txt'
    val_record_file = 'val_jpg.txt'

    accuracy(train_record_file=train_record_file,
             val_record_file=val_record_file,
             labels_nums=labels_nums,
             data_shape=data_shape,
          )
