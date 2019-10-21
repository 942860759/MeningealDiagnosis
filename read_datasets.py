# -*- coding:utf-8 _*-   
# author: tangwei 
# time: 2018/09/20 10：39

'''
    给测试集和验证集 创建tfrecord格式的文件
'''

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import pydicom


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


def show_image(title, image):
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


def load_labels_file(filename,shuffle=False):
    '''
    载图txt文件，文件中每行为一个图片信息，且以空格隔开：图像路径 标签1 标签2，如：test_image/1.jpg 0 2
    :param filename:
    :param labels_num :labels个数
    :param shuffle :是否打乱顺序
    :return:images type->list
    :return:labels type->list
    '''
    image_labels = []
    with open(filename) as f:
        lines_list = f.readlines()
        if shuffle:
            random.shuffle(lines_list)

        count = 0
        for i,lines in enumerate(lines_list):
            if i == 0:
                continue

            line = lines.rstrip().split(',')
            image_list = line[0].split('_')
            image_name = image_list[0] + '_' + image_list[1]
            label_name = image_list[2]

            if count % 6 == 0:
                dict = {'image': image_name, 'epidural': 0, 'intraparenchymal': 0,
                        'intraventricular': 0, 'subarachnoid': 0, 'subdural': 0, 'any': 0 }

            dict[label_name] = line[1]


            count += 1
            if count % 6 == 0:
                count = 0
                image_labels.append(dict)

    return image_labels


def read_image(filename, resize_height, resize_width, normalization=False):
    '''
    读取图片数据,默认返回的是uint8,[0,255]
    :param filename:
    :param resize_height:
    :param resize_width:
    :param normalization:是否归一化到[0.,1.0]
    :return: 返回的图片数据
    '''

    ds = pydicom.read_file(filename)  # 读取.dcm文件
    bgr_image = ds.pixel_array  # 提取图像信息

    # show_image('test', img)

    # bgr_image = cv2.imread(filename)
    '''
    flag=-1时，8位深度，原通道
    flag=0，8位深度，1通道
    flag=1, 8位深度，3通道
    flag=2，原深度，1通道
    flag=3, 原深度，3通道
    flag=4，8位深度，3通道
    '''
    #     if len(bgr_image.shape)==2:#若是灰度图则转为三通道
    #         print("Warning:gray image",filename)
    #         bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)
    #
    #     rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)#将BGR转为RGB

    if len(bgr_image.shape) == 3:#若是RGB则转为灰度图
        print("Warning:RGB image", filename)
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    rgb_image = bgr_image
    #show_image(filename,rgb_image)
    # rgb_image=Image.open(filename)
    if resize_height > 0 and resize_width > 0:
        rgb_image = cv2.resize(rgb_image, (resize_width,resize_height))
    rgb_image = np.asanyarray(rgb_image)
    if normalization:
        # 不能写成:rgb_image=rgb_image/255
        rgb_image = rgb_image/255.0
    # show_image("src resize image",rgb_image)
    rgb_image = np.expand_dims(rgb_image, 2)  # 增加维度500*500*1
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
    # sexs = tf.one_hot(sexs, sexs_nums, 1, 0)
    # tf.one_hot()函数规定输入的元素indices从0开始，最大的元素值不能超过（depth - 1），
    # 因此能够表示（depth + 1）个单位的输入。若输入的元素值超出范围，输出的编码均为 [0, 0 … 0, 0]。
    if shuffle:
        images_batch, labels_batch = tf.train.shuffle_batch([images,labels],
                                                           batch_size=batch_size,
                                                           capacity=capacity,
                                                           min_after_dequeue=min_after_dequeue,
                                                           num_threads=num_threads)
    else:
        images_batch, labels_batch= tf.train.batch([images,labels],
                                                    batch_size=batch_size,
                                                    capacity=capacity,
                                                    num_threads=num_threads)
    return images_batch, labels_batch


# 将数据集按比例随机分为训练集和测试集
def trainTestSplit(trainingSet, trainingLabels, train_size):
    totalNum = int(len(trainingSet))
    trainIndex = list(range(totalNum))  # 存放训练集的下标

    x_test = []  # 存放测试集输入
    y_test = []  # 存放测试集输出

    x_train = []  # 存放训练集输入
    y_train = []  # 存放训练集输出

    trainNum = int(totalNum * train_size)  # 划分训练集的样本数
    for i in range(trainNum):
        randomIndex = int(np.random.uniform(0, len(trainIndex)))
        x_test.append(trainingSet[trainIndex[randomIndex]])
        y_test.append(trainingLabels[trainIndex[randomIndex]])
        del (trainIndex[randomIndex])  # 删除已经放入测试集的下标

    for i in range(totalNum - trainNum):
        x_train.append(trainingSet[trainIndex[i]])
        y_train.append(trainingLabels[trainIndex[i]])

    return x_train, x_test, y_train, y_test


def read_data(file, train_image, resize_height, resize_width):
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

    # 加载文件,仅获取一个label
    images_labels = load_labels_file(file, shuffle=False)

    images = []
    labels = []

    for item in images_labels:
        image_name = item['image']
        image_path = train_image + '\\' + image_name + '.dcm'

        images.append(image_path)

        # gry_image = read_image(image_path, resize_height, resize_width, normalization=True)
        # images.append(gry_image)

        labels.append([item['epidural'], item['intraparenchymal'], item['intraventricular'], item['subarachnoid'], item['subdural'], item['any'] ])

    return images, labels



if __name__ == '__main__':
    # 参数设置

    resize_height = 512  # 指定存储图片高度
    resize_width = 512  # 指定存储图片宽度
    shuffle = False
    log = 5

    # train_csv ='E:\\download\\archive\\stage_1_train.csv'  # csv路径
    # train_image = 'E:\\download\\archive\\stage_1_train_images'
    #
    # total_images, total_labels = read_data(train_csv, train_image, resize_height, resize_width)
    #
    # train_images, val_images, train_labels, val_labels = trainTestSplit(total_images, total_labels, train_size=0.002)
    #
    # train_files_list = []
    # for i, [image_name, labels] in enumerate(zip(train_images, train_labels)):
    #     image_path = image_name
    #     train_files_list.append([image_path] + [label for label in labels])
    #
    # val_files_list = []
    # for i, [image_name, labels] in enumerate(zip(val_images, val_labels)):
    #     image_path = image_name
    #     val_files_list.append([image_path] + [label for label in labels])
    #
    # train_quantity_txt = 'train.txt'  # 训练数据路径
    # val_quantity_txt = 'val.txt'  # 验证测试数据路径
    #
    # write_txt(train_files_list, train_quantity_txt, mode='w')
    # write_txt(val_files_list, val_quantity_txt, mode='w')

    test_csv = 'H:\\archive\\stage_1_sample_submission.csv'  # csv路径
    test_image = 'H:\\archive\\stage_1_test_images'
    test_images, test_labels = read_data(test_csv, test_image, resize_height, resize_width)
    test_files_list = []
    for i, [image_name, labels] in enumerate(zip(test_images, test_labels)):
        image_path = image_name
        test_files_list.append([image_path] + [label for label in labels])
    test_quantity_txt = 'test.txt'  # 测试数据路径
    write_txt(test_files_list, test_quantity_txt, mode='w')

