# -*- coding:utf-8 _*-   
# author: tangwei 
# time: 2018/09/20 18：20
'''
    预测一张图片
'''
#coding=utf-8

import tensorflow as tf
import numpy as np
import os
import ihnet_v2
import create_tf_record
import tensorflow.contrib.slim as slim

# 定义dropout的概率
keep_prob = tf.placeholder(tf.float32,name='keep_prob')
is_training = tf.placeholder(tf.bool, name='is_training')


def stats_graph(graph):
  flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
  params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
  print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))


def write_txt(content, filename, mode='w'):
    """保存txt数据
    :param content:需要保存的数据,type->list
    :param filename:文件名
    :param mode:读写模式:'w' or 'a'
    :return: void
    """
    with open(filename, mode) as f:
        for line in content:
            str_line = ""
            for col, data in enumerate(line):
                if not col == len(line) - 1:
                    # 以空格作为分隔符
                    str_line = str_line + str(data) + " "
                else:
                    # 每行最后一个数据用换行符“\n”
                    str_line = str_line + str(data) + "\n"
            f.write(str_line)


def predict(models_path, image_path, labels_nums, data_format):
    [resize_height, resize_width, depths] = data_format

    input_images = tf.placeholder(dtype=tf.float32, shape=[None, resize_height, resize_width, depths], name='input')

    with slim.arg_scope(ihnet_v2.ihnet_arg_scope()):
        out, end_points = ihnet_v2.ihnet(inputs=input_images,
                                         num_classes=labels_nums,
                                         dropout_keep_prob=keep_prob,
                                         is_training=is_training)

    # 将输出结果进行softmax分布,再求最大概率所属类别
    class_id = out

    saver = tf.train.Saver()
    sess = tf.InteractiveSession()

    graph = sess.graph
    stats_graph(graph)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    saver.restore(sess, models_path)

    files_list = []

    if not os.path.exists(image_path):
        print('Err:no image', image_path)
        return
    im = create_tf_record.read_image(image_path, resize_height, resize_width)
    im = im[np.newaxis, :]
    # 增加维度 1*500*500*1
    im = im.astype(np.float32)
    # 犯了一个错误，labels_list[0] 导致sex总是为1

    pre_label = sess.run([class_id],
                          feed_dict={input_images: im,
                                     keep_prob: 1,
                                     is_training: False})
    print("{} is: pre labels:{}".format(image_path, pre_label))
    files_list.append([image_path, pre_label])

    write_txt(files_list, 'test/one_predict.txt', mode='w')
    sess.close()


if __name__ == '__main__':

    class_nums = 6
    image_path = 'demo.dcm'
    models_path = 'models/best_models_472500_0.7792.ckpt'

    resize_height = 512  # 指定存储图片高度
    resize_width = 512  # 指定存储图片宽度
    depths = 3
    data_format = [resize_height, resize_width, depths]
    predict(models_path, image_path, class_nums, data_format)
