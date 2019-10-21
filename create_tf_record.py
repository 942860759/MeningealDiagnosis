# -*- coding:utf-8 _*-   
# author: tangwei 
# time: 2018/09/20 10：39

'''
    给测试集和验证集 创建tfrecord格式的文件
'''

import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import random
import pydicom
import numpy

##########################################################################
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
# 生成字符串型的属性


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
# 生成实数型的属性


def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def get_example_nums(tf_records_filenames):
    '''
    统计tf_records图像的个数(example)个数
    :param tf_records_filenames: tf_records文件路径
    :return:
    '''
    nums= 0
    for record in tf.python_io.tf_record_iterator(tf_records_filenames):
        nums += 1
    return nums


def get_txt_nums(filenames):
    '''
    统计tf_records图像的个数(example)个数
    :param tf_records_filenames: tf_records文件路径
    :return:
    '''
    with open(filenames) as f:
        lines_list = f.readlines()
    nums = len(lines_list)
    return nums


def _normalize(x):
    x_max = x.max()
    x_min = x.min()
    if x_max != x_min:
        z = (x - x_min) / (x_max - x_min)
        return z
    return np.zeros(x.shape)


def _read(path, desired_size):
    """Will be used in DataGenerator"""

    dcm = pydicom.dcmread(path)

    slope, intercept = dcm.RescaleSlope, dcm.RescaleIntercept

    try:
        img = np.clip((dcm.pixel_array * slope + intercept), -50, 450)
    except:
        img = np.zeros(desired_size[:2])

    if img.shape != desired_size[:2]:
        img = cv2.resize(img, desired_size[:2], interpolation=cv2.INTER_LINEAR)

    img = _normalize(img)

    return np.stack((img,) * 3, axis=-1)


def show_image(title,image):
    '''
    显示图片
    :param title: 图像标题
    :param image: 图像的数据
    :return:
    '''
    # plt.figure("show_image")
    # print(image.dtype)
    plt.imshow(image)
    plt.axis('on')    # 关掉坐标轴为 off
    plt.title(title)  # 图像题目
    plt.show()


def load_labels_file(filename, labels_num=1, shuffle=False):
    '''
    载图txt文件，文件中每行为一个图片信息，且以空格隔开：图像路径 标签1 标签2，如：test_image/1.jpg 0 2
    :param filename:
    :param labels_num :labels个数
    :param shuffle :是否打乱顺序
    :return:images type->list
    :return:labels type->list
    '''
    images = []
    labels = []
    with open(filename) as f:
        lines_list = f.readlines()
        if shuffle:
            random.shuffle(lines_list)

        for lines in lines_list:
            line = lines.rstrip().split(' ')
            label = []
            for i in range(labels_num):
                label.append(float(line[i+1]))
            images.append(line[0])
            labels.append(label)
    return images, labels


def read_image(filename, resize_height, resize_width):
    '''
    读取图片数据,默认返回的是uint8,[0,255]
    :param filename:
    :param resize_height:
    :param resize_width:
    :param normalization:是否归一化到[0.,1.0]
    :return: 返回的图片数据
    '''

    rgb_image = _read(filename, (resize_height, resize_width, 3))  # 提取图像信息

    rgb_image = (rgb_image * 255).astype(np.uint8)  # 转换为0--256的灰度uint8类型
    # print(rgb_image.shape)
    # show_image('test', rgb_image)
    # cv2.imwrite('dicom.jpg', rgb_image)

    '''
    flag=-1时，8位深度，原通道
    flag=0，8位深度，1通道
    flag=1, 8位深度，3通道
    flag=2，原深度，1通道
    flag=3, 原深度，3通道
    flag=4，8位深度，3通道
    '''

    return rgb_image


def get_batch_images(images, labels, batch_size, shuffle=False, num_threads=1):
    '''
    :param images:图像
    :param labels:标签
    :param batch_size:
    :param labels_nums:标签个数
    :param one_hot:是否将labels转为one_hot的形式
    :param shuffle:是否打乱顺序,一般train时shuffle=True,验证时shuffle=False
    :return:返回batch的images和labels
    '''
    min_after_dequeue = 200
    capacity = min_after_dequeue + 3 * batch_size  # 保证capacity必须大于min_after_dequeue参数值

    if shuffle:
        images_batch, labels_batch = tf.train.shuffle_batch([images, labels],
                                                             batch_size=batch_size,
                                                             capacity=capacity,
                                                             min_after_dequeue=min_after_dequeue,
                                                             num_threads=num_threads)
    else:
        images_batch, labels_batch= tf.train.batch([images, labels],
                                                    batch_size=batch_size,
                                                    capacity=capacity,
                                                    num_threads=num_threads)

    return images_batch, labels_batch


def read_records(filename, resize_height, resize_width, type=None):
    '''
    解析record文件:源文件的图像数据是RGB,uint8,[0,255],一般作为训练数据时,需要归一化到[0,1]
    :param filename:
    :param resize_height:
    :param resize_width:
    :param type:选择图像数据的返回类型
         None:默认将uint8-[0,255]转为float32-[0,255]
         normalization:归一化float32-[0,1]
         centralization:归一化float32-[0,1],再减均值中心化
    :return:
    '''
    # 创建文件队列,不限读取的数量
    filename_queue = tf.train.string_input_producer([filename])
    # create a reader from file queue
    reader = tf.TFRecordReader()
    # reader从文件队列中读入一个序列化的样本
    _, serialized_example = reader.read(filename_queue)
    # get feature from serialized example
    # 解析符号化的样本
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label_epidural': tf.FixedLenFeature([], tf.int64),
            'label_intraparenchymal': tf.FixedLenFeature([], tf.int64),
            'label_intraventricular': tf.FixedLenFeature([], tf.int64),
            'label_subarachnoid': tf.FixedLenFeature([], tf.int64),
            'label_subdural': tf.FixedLenFeature([], tf.int64),
            'label_any': tf.FixedLenFeature([], tf.int64),
        }
    )
    tf_image = tf.decode_raw(features['image_raw'], tf.uint8)#获得图像原始的数据
    tf_image = tf.reshape(tf_image, [resize_height, resize_width, 3])  # 设置图像的维度

    tf_label_epidural = tf.cast(features['label_epidural'], tf.float32)
    tf_label_intraparenchymal = tf.cast(features['label_intraparenchymal'], tf.float32)
    tf_label_intraventricular = tf.cast(features['label_intraventricular'], tf.float32)
    tf_label_subarachnoid = tf.cast(features['label_subarachnoid'], tf.float32)
    tf_label_subdural = tf.cast(features['label_subdural'], tf.float32)
    tf_label_any = tf.cast(features['label_any'], tf.float32)

    tf_label = [tf_label_epidural, tf_label_intraparenchymal, tf_label_intraventricular, tf_label_subarachnoid, tf_label_subdural, tf_label_any]

    # 存储的图像类型为uint8,tensorflow训练时数据必须是tf.float32
    if type is None:
        tf_image = tf.cast(tf_image, tf.float32)
    elif type=='normalization':# [1]若需要归一化请使用:
        # 仅当输入数据是uint8,才会归一化[0,255]
        tf_image = tf.cast(tf_image, tf.float32) * (1. / 255.0)  # 归一化
    elif type=='centralization':
        # 若需要归一化,且中心化,假设均值为0.5,请使用:
        tf_image = tf.cast(tf_image, tf.float32) * (1. / 255) - 0.5  # 中心化

    return tf_image, tf_label


def create_records(file, output_record_dir, resize_height, resize_width, shuffle, log=5):
    '''
    实现将图像原始数据,label,长,宽等信息保存为record文件
    注意:读取的图像数据默认是uint8,再转为tf的字符串型BytesList保存,解析请需要根据需要转换类型
    :param file:输入保存图片信息的txt文件(image_dir+file构成图片的路径)
    :param image_dir:输入保存图片的文件(image_dir)
    :param output_record_dir:保存record文件的路径
    :param resize_height:
    :param resize_width:
    PS:当resize_height或者resize_width=0是,不执行resize
    :param shuffle:是否打乱顺序
    :param log:log信息打印间隔
    '''
    # 加载文件,仅获取一个label
    images_list, labels_list=load_labels_file(file, 6, shuffle)

    writer = tf.python_io.TFRecordWriter(output_record_dir)
    for i, [image_name, labels] in enumerate(zip(images_list, labels_list)):

        image_path = images_list[i]
        if not os.path.exists(image_path):
            print('Err:no image',image_path)
            continue
        image = read_image(image_path, resize_height, resize_width)
        image_raw = image.tostring()
        if i%log==0 or i==len(images_list)-1:
            print('------------processing:%d-th------------' % (i))
            print('current image_path=%s' % (image_path),'shape:{}'.format(image.shape),'labels:{}'.format(labels))
        # 这里仅保存一个label,多label适当增加"'label': _int64_feature(label)"项

        label_epidural = labels[0]
        label_intraparenchymal = labels[1]
        label_intraventricular = labels[2]
        label_subarachnoid = labels[3]
        label_subdural = labels[4]
        label_any = labels[5]

        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': _bytes_feature(image_raw),
            'label_epidural': _int64_feature(label_epidural),
            'label_intraparenchymal': _int64_feature(label_intraparenchymal),
            'label_intraventricular': _int64_feature(label_intraventricular),
            'label_subarachnoid': _int64_feature(label_subarachnoid),
            'label_subdural': _int64_feature(label_subdural),
            'label_any': _int64_feature(label_any),
        }))
        writer.write(example.SerializeToString())
    writer.close()


def disp_records(record_file, resize_height, resize_width, show_nums=4):
    '''
    解析record文件，并显示show_nums张图片，主要用于验证生成record文件是否成功
    :param tfrecord_file: record文件路径
    :return:
    '''
    # 读取record函数
    tf_image, tf_label = read_records(record_file, resize_height, resize_width, type='None')
    # tf_image = tf.squeeze(tf_image, 2)#降维500*500
    # 显示前4个图片
    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(show_nums):
            image, label= sess.run([tf_image,tf_label])  # 在会话中取出image和label
            # image = tf_image.eval()
            # 直接从record解析的image是一个向量,需要reshape显示
            # image = image.reshape([height,width,depth])
            print('shape:{},tpye:{},labels:{}'.format(image.shape,image.dtype,label))
            # pilimg = Image.fromarray(np.asarray(image_eval_reshape))
            # pilimg.show()
            show_image("image:%s"%(label),image)
        coord.request_stop()
        coord.join(threads)


def batch_test(record_file,resize_height, resize_width):
    '''
    :param record_file: record文件路径
    :param resize_height:
    :param resize_width:
    :return:
    :PS:image_batch, label_batch一般作为网络的输入
    '''
    # 读取record函数
    tf_image, tf_label = read_records(record_file,resize_height,resize_width,type='None')
    image_batch, label_batch= get_batch_images(tf_image,
                                               tf_label,
                                               batch_size=4,
                                               shuffle=False)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:  # 开始一个会话
        sess.run(init)
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(20):
            # 在会话中取出images和labels
            images, labels = sess.run([image_batch, label_batch])
            # 这里仅显示每个batch里第一张图片
            show_image("image", images[0, :, :, :])
            print('shape:{},tpye:{},labels:{}'.format(images.shape,images.dtype,labels))

        # 停止所有线程
        coord.request_stop()
        coord.join(threads)


def read_from_files(filename, resize_height, resize_width, shuffle=False, type = 'normalization'):
    '''
    读取图片数据,默认返回的是uint8,[0,255]
    :param filename:
    :param resize_height:
    :param resize_width:
    :param normalization:是否归一化到[0.,1.0]
    :return: 返回的图片数据
    '''
    with open(filename) as f:
        lines_list = f.readlines()
        lines_list = [ line.rstrip() for line in lines_list]
        if shuffle:
            random.shuffle(lines_list)

    path = tf.convert_to_tensor(lines_list, dtype=tf.string)
    valuequeue = tf.train.string_input_producer(path, shuffle=shuffle, num_epochs=4)
    value = valuequeue.dequeue()
    dir, label_epidural, label_intraparenchymal, label_intraventricular, label_subarachnoid, label_subdural, label_any  = tf.decode_csv(records=value,
                                                                        record_defaults=[["string"], [""], [""], [""], [""], [""], [""]], field_delim=" ")

    label_epidural = tf.string_to_number(label_epidural, tf.int32)
    label_intraparenchymal = tf.string_to_number(label_intraparenchymal, tf.int32)
    label_intraventricular = tf.string_to_number(label_intraventricular, tf.int32)
    label_subarachnoid = tf.string_to_number(label_subarachnoid, tf.int32)
    label_subdural = tf.string_to_number(label_subdural, tf.int32)
    label_any = tf.string_to_number(label_any, tf.int32)
    tf_label = [label_epidural, label_intraparenchymal, label_intraventricular, label_subarachnoid,
                label_subdural, label_any]

    imagecontent = tf.read_file(dir)
    tf_image = tf.image.decode_png(imagecontent, channels=3, dtype=tf.uint8)
    tf_image = tf.image.resize_images(tf_image, [resize_height, resize_width])
    tf_image = tf.reshape(tf_image, [resize_height, resize_width, 3])  # 设置图像的维度

    # 存储的图像类型为uint8,tensorflow训练时数据必须是tf.float32
    if type is None:
        tf_image = tf.cast(tf_image, tf.float32)
    elif type == 'normalization':  # [1]若需要归一化请使用:
        # 仅当输入数据是uint8,才会归一化[0,255]
        tf_image = tf.cast(tf_image, tf.float32) * (1. / 255.0)  # 归一化
    elif type == 'centralization':
        # 若需要归一化,且中心化,假设均值为0.5,请使用:
        tf_image = tf.cast(tf_image, tf.float32) * (1. / 255) - 0.5  # 中心化

    return tf_image, tf_label


def batch_file_test(filename, resize_height, resize_width):
    '''
    :param record_file: record文件路径
    :param resize_height:
    :param resize_width:
    :return:
    :PS:image_batch, label_batch一般作为网络的输入
    '''
    # 读取record函数
    tf_image, tf_label = read_from_files(filename, resize_height, resize_width, shuffle=True, type='normalization')
    image_batch, label_batch= get_batch_images(tf_image,
                                               tf_label,
                                               batch_size=4,
                                               shuffle=False)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:  # 开始一个会话
        sess.run(init)
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            for i in range(5):
                # 在会话中取出images和labels
                images, labels = sess.run([image_batch, label_batch])
                # 这里仅显示每个batch里第一张图片
                show_image("image", images[0, :, :, :])
                print('shape:{},tpye:{},labels:{}'.format(images.shape, images.dtype, labels))
        except tf.errors.OutOfRangeError:
            print('done')
        finally:
            coord.request_stop()
        # 停止所有线程
        # coord.request_stop()
        coord.join(threads)


def change_jpg(filename, resize_height, resize_width):
    '''
    E:\download\archive\stage_1_train_images\ID_7e911671e.dcm
    :param filename:
    :param outname:
    :param resize_height:
    :param resize_width:
    :return:
    '''
    images_list, labels_list = load_labels_file(filename, 6, shuffle)
    for i, [image_name, labels] in enumerate(zip(images_list, labels_list)):

        # image_path = 'H:\\' + images_list[i].split('\\', 2 )[-1]
        image_path = 'H:\\archive\\' + images_list[i].split('\\', 2 )[-1]
        if not os.path.exists(image_path):
            print('Err:no image',image_path)
            continue
        image = read_image(image_path, resize_height, resize_width)

        # save_path = images_list[i].split('.', 1 )[0] + '.jpg'
        save_path = 'E:\\\download\\archive\\stage_1_test_images\\' + images_list[i].split('.', 1 )[0].split('\\')[-1] + '.jpg'

        cv2.imwrite(save_path, image)


def change_jpg_txt(filename_train='train.txt', filename_val='val.txt', filename_test='test.txt', shuffle=False):
    '''
    E:\download\archive\stage_1_train_images\ID_7e911671e.dcm
    :param filename:
    :param outname:
    :param resize_height:
    :param resize_width:
    :return:
    '''
    train_images, train_labels = load_labels_file(filename_train, 6, shuffle)
    train_files_list = []
    for i, [image_name, labels] in enumerate(zip(train_images, train_labels)):
        image_path = image_name.split('.', 1)[0] + '.jpg'
        train_files_list.append([image_path] + [int(label) for label in labels])
    train_quantity_txt = 'train_jpg.txt'  # 训练数据路径
    write_txt(train_files_list, train_quantity_txt, mode='w')

    val_images, val_labels = load_labels_file(filename_val, 6, shuffle)
    val_files_list = []
    for i, [image_name, labels] in enumerate(zip(val_images, val_labels)):
        image_path = image_name.split('.', 1 )[0] + '.jpg'
        val_files_list.append([image_path] + [int(label) for label in labels])
    val_quantity_txt = 'val_jpg.txt'  # 验证测试数据路径
    write_txt(val_files_list, val_quantity_txt, mode='w')

    test_images, test_labels = load_labels_file(filename_test, 6, shuffle)
    test_files_list = []
    for i, [image_name, labels] in enumerate(zip(test_images, test_labels)):
        image_path = 'E:\\download\\' + image_name.split('.', 1)[0].split('\\', 1)[-1] + '.jpg'
        test_files_list.append([image_path] + [int(label) for label in labels])
    test_quantity_txt = 'test_jpg.txt'  # 验证测试数据路径
    write_txt(test_files_list, test_quantity_txt, mode='w')


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


if __name__ == '__main__':
    # 参数设置

    resize_height = 512  # 指定存储图片高度
    resize_width = 512  # 指定存储图片宽度
    shuffle = False
    log = 5

    # # # 产生train.record文件
    # train_labels = 'train.txt'  # 图片路径
    # # train_record_output = 'D:\\tfrecord\\train.tfrecords'
    # train_record_output = 'H:\\train.tfrecords'
    # create_records(train_labels, train_record_output, resize_height, resize_width, shuffle, log)
    # train_nums = get_example_nums(train_record_output)
    # print("save train example nums={}".format(train_nums))

    # # 产生val.record文件
    # val_labels = 'val.txt'  # 图片路径
    # val_record_output = 'D:\\tfrecord\\val.tfrecords'
    # create_records(val_labels, val_record_output, resize_height, resize_width, shuffle, log)
    # val_nums = get_example_nums(val_record_output)
    # print("save val example nums={}".format(val_nums))


    # 测试显示函数
    # disp_records(val_record_output, resize_height, resize_width)
    # batch_test(val_record_output, resize_height, resize_width)

    # image_path = 'boneage/images-predict/3.jpg'
    # image_path = 'boneage/boneage-ownership-dataset/1377.png'
    # if not os.path.exists(image_path):
    #     print('Err:no image', image_path)
    # im = read_image(image_path, resize_height, resize_width, normalization=True)

    # filename = 'train.txt'
    # tf_image = read_from_files(filename, resize_height, resize_width, shuffle=False, type='normalization')

    # change_jpg_txt()

    # change_jpg('train.txt', resize_height, resize_width)
    # change_jpg('val.txt', resize_height, resize_width)

    # change_jpg('test.txt', resize_height, resize_width)

    batch_file_test('val_jpg.txt', resize_height, resize_width)