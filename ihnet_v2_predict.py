# -*- coding:utf-8 _*-   
# author: tangwei 
# time: 2018/09/20 18：20
'''
    预测结果
'''
#coding=utf-8

import tensorflow as tf
import ihnet_v2
import sys
from create_tf_record import *
import tensorflow.contrib.slim as slim
import pandas as pd


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


def load_labels_file(filename, labels_num=2, shuffle=False):
    '''
    载图txt文件，文件中每行为一个图片信息，且以空格隔开：图像路径 标签1 标签2，如：test_image/1.jpg 0 2
    :param filename:
    :param labels_num :labels个数
    :param shuffle :是否打乱顺序
    :return:images type->list
    :return:labels type->list
    '''
    images=[]
    labels=[]
    with open(filename) as f:
        lines_list=f.readlines()
        if shuffle:
            random.shuffle(lines_list)

        for lines in lines_list:
            line=lines.rstrip().split(' ')
            label=[]
            for i in range(labels_num):
                label.append(int(line[i+1]))
            images.append(line[0])
            labels.append(label)
    return images,labels


def predict(models_path, test_txt, labels_nums, data_format):
    [resize_height, resize_width, depths] = data_format

    input_images = tf.placeholder(dtype=tf.float32, shape=[None, resize_height, resize_width, depths], name='input')
    # 定义dropout的概率
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    is_training = tf.placeholder(tf.bool, name='is_training')

    with slim.arg_scope(ihnet_v2.ihnet_arg_scope()):
        out, end_points = ihnet_v2.ihnet(inputs=input_images,
                                         num_classes=labels_nums,
                                         dropout_keep_prob=1.0,
                                         is_training=False)

    # 将输出结果进行softmax分布,再求最大概率所属类别
    class_id = out

    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    saver.restore(sess, models_path)

    files_list = []
    files_list_post = []
    # 加载文件,仅获取一个label
    images_list, labels_list = load_labels_file(test_txt, 2, shuffle=False)

    for i, [image_name, labels] in enumerate(zip(images_list, labels_list)):

        image_path = image_name
        if not os.path.exists(image_path):
            print('Err:no image', image_path)
            continue

        bgr_image = cv2.imread(image_path)
        if len(bgr_image.shape) == 2:  # 若是RGB则转为灰度图
            print("Warning:RGB image", image_path)
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)
        else:
            rgb_image = bgr_image
        if resize_height > 0 and resize_width > 0:
            rgb_image = cv2.resize(rgb_image, (resize_width, resize_height))
        rgb_image = np.asanyarray(rgb_image)
        im = rgb_image / 255.0

        im = im[np.newaxis, :]   # 增加维度1*500*500*1

        im = im.astype(np.float32)

        pre_labels = sess.run([class_id], feed_dict={input_images: im,
                                                    keep_prob: 1,
                                                    is_training: False})
        pre_label = pre_labels[0][0]
        print("{} is: pre labels:{}".format(image_path, pre_label))
        files_list.append([image_path] + [label for label in pre_label])

        name = image_name.split('.', 1)[0].split('\\')[-1]
        files_list_post.append([name + '_epidural', pre_label[0]])
        files_list_post.append([name + '_intraparenchymal', pre_label[1]])
        files_list_post.append([name + '_intraventricular', pre_label[2]])
        files_list_post.append([name + '_subarachnoid', pre_label[3]])
        files_list_post.append([name + '_subdural', pre_label[4]])
        files_list_post.append([name + '_any', pre_label[5]])

    datas = np.array(files_list_post)
    ##写入文件
    pd_data = pd.DataFrame(datas, columns=['ID', 'Lable'])

    pd_data.to_csv('test/submit_predict.csv')
    write_txt(files_list, 'test/test_predict.txt', mode='w')
    sess.close()


if __name__ == '__main__':

    class_nums = 6
    test_txt = 'test_jpg.txt'
    ckpt = tf.train.get_checkpoint_state('./logs/')
    if ckpt and ckpt.model_checkpoint_path:
        print(ckpt.model_checkpoint_path)
        models_path = ckpt.model_checkpoint_path
        # models_path = 'models/RSNA/model.ckpt-133800'
    else:
        print('No checkpoint!')
        sys.exit(0)

    resize_height = 512  # 指定存储图片高度
    resize_width = 512  # 指定存储图片宽度
    depths = 3
    data_format = [resize_height, resize_width, depths]
    predict(models_path, test_txt, class_nums, data_format)
