# -*- coding:utf-8 _*-
import tensorflow as tf
from datetime import datetime
import math
import time
from CBAM import *

# 如果想冻结所有层，可以指定slim.conv2d中的 ,trainable=False
slim = tf.contrib.slim
# Slim is an interface to contrib functions, examples and models.
# 只是一个接口作用
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

# 匿名函数 lambda x: x * x  实际上就是：返回x的平方
# tf.truncated_normal_initializer产生截断的正态分布

def PReLU(_x, scope=None):
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        _alpha = tf.get_variable("prelu", shape=_x.get_shape()[-1],
                                 dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)


# 默认参数：卷积的激活函数、权重初始化方式、标准化器等
def ihnet_arg_scope(weight_decay=0.00004,  # 设置L2正则的weight_decay 0.00004
                    # weight decay的作用是调节模型复杂度对损失函数的影响，若weight decay很大，则复杂的模型损失函数的值也就大
                    stddev=0.1,  # 标准差默认值0.1
                    batch_norm_var_collection='moving_vars'):
    # 定义batch normalization（批量标准化/归一化）的参数字典
    batch_norm_params = {
        # 'decay': 0.9900,  # 定义参数衰减系数
        'decay': 0.9997,  # 定义参数衰减系数
        # 该参数能够衡量使用指数衰减函数更新均值方差时，更新的速度，取值通常在0.999-0.99-0.9之间，值
        # 越小，代表更新速度越快，而值太大的话，有可能会导致均值方差更新太慢，而最后变成一个常量1，而
        # 这个值会导致模型性能较低很多.另外，如果出现过拟合时，也可以考虑增加均值和方差的更新速度，也
        # 就是减小decay
        'epsilon': 0.001,  # 就是在归一化时，除以方差时，防止方差为0而加上的一个数
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        # force in-place updates of mean and variance estimates
        # 该参数有一个默认值，ops.GraphKeys.UPDATE_OPS，当取默认值时，slim会在当前批训练完成后再更新均
        # 值和方差，这样会存在一个问题，就是当前批数据使用的均值和方差总是慢一拍，最后导致训练出来的模
        # 型性能较差。所以，一般需要将该值设为None，这样slim进行批处理时，会对均值和方差进行即时更新，
        # 批处理使用的就是最新的均值和方差。

        # 另外，不论是即使更新还是一步训练后再对所有均值方差一起更新，对测试数据是没有影响的，即测试数
        # 据使用的都是保存的模型中的均值方差数据，但是如果你在训练中需要测试，而忘了将is_training这个值
        # 改成false，那么这批测试数据将会综合当前批数据的均值方差和训练数据的均值方差。而这样做应该是不
        # 正确的。
        'variables_collections': {
            'beta': None,
            'gamma': None,
            'moving_mean': [batch_norm_var_collection],
            'moving_variance': [batch_norm_var_collection],  # 值就是前面设置的batch_norm_var_collection='moving_vars'
        }
    }

    # 给函数的参数自动赋予某些默认值
    # slim.arg_scope常用于为tensorflow里的layer函数提供默认值以使构建模型的代码更加紧凑苗条(slim):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l1_l2_regularizer(scale_l1=0.000001, scale_l2=weight_decay)):
        # 对[slim.conv2d, slim.fully_connected]自动赋值，可以是列表或元组
        # 使用slim.arg_scope后就不需要每次都重复设置参数了，只需要在有修改时设置
        with slim.arg_scope(  # 嵌套一个slim.arg_scope对卷积层生成函数slim.conv2d的几个参数赋予默认值
                [slim.conv2d],
                weights_initializer=slim.variance_scaling_initializer(),  # 权重初始化器
                # weights_initializer=trunc_normal(stddev),  # 权重初始化器
                activation_fn=tf.nn.relu,  # 激活函数
                # activation_fn=PReLU,  # 激活函数
                normalizer_fn=slim.batch_norm,  # 标准化器
                normalizer_params=batch_norm_params) as sc:  # 标准化器的参数设置为前面定义的batch_norm_params
            return sc  # 最后返回定义好的scope


def ihnet_base(inputs, scope=None):
    '''
    Args:
    inputs：输入的tensor
    scope：包含了函数默认参数的环境
    '''
    end_points = {}  # 定义一个字典表保存某些关键节点供之后使用

    with tf.variable_scope(scope, 'IhnetV2', [inputs]):
        end_points['origin_pic'] = inputs

        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],  # 对三个参数设置默认值
                            stride=1, padding='VALID'):
            # 输入图像尺寸 512 x 512 x 1
            # slim.conv2d的第一个参数为输入的tensor，第二个是输出的通道数，卷积核尺寸，步长stride，padding模式
            net = slim.conv2d(inputs, 32, [3, 3], stride=2, scope='Conv2d_1a_3x3')
            # 输出尺寸 255 x 255 x 32
            net = slim.conv2d(net, 32, [3, 3], scope='Conv2d_1b_3x3')
            # 输出尺寸 253 x 253 x 32
            net = slim.conv2d(net, 64, [3, 3], padding='SAME', scope='Conv2d_1c_3x3')
            # 输出尺寸 253 x 235 x 64
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='MaxPool_1a_3x3')
            # 输出尺寸 126 x 126 x 64
            net = cbam_block_channel_first(net, scope='CBAM_Block_Channel_First_1')
            # cbam attention

            net = slim.conv2d(net, 80, [1, 1], scope='Conv2d_2a_1x1')
            # 输出尺寸 126 x 126 x 80.
            net = slim.conv2d(net, 192, [3, 3], scope='Conv2d_2a_3x3')
            # 输出尺寸 124 x 124 x 192.
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='MaxPool_2a_3x3')
            # 输出尺寸 61 x 61 x 192.
            net = cbam_block_channel_first(net, scope='CBAM_Block_Channel_First_2')
            # cbam attention

            with tf.variable_scope('ResBlock1'):
                end_point = 'Conv2d_3a_1x1'
                with slim.arg_scope([slim.conv2d], stride=1, padding='SAME'):
                    with tf.variable_scope('BlockInceptionA'):
                        with tf.variable_scope('Branch_0'):
                            branch_0 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
                        with tf.variable_scope('Branch_1'):
                            branch_1 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
                            branch_1 = slim.conv2d(branch_1, 32, [3, 3], scope='Conv2d_0b_3x3')
                        with tf.variable_scope('Branch_2'):
                            branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
                            branch_2 = slim.conv2d(branch_2, 48, [3, 3], scope='Conv2d_0b_3x3')
                            branch_2 = slim.conv2d(branch_2, 64, [3, 3], scope='Conv2d_0c_3x3')
                        residual = tf.concat(axis=3, values=[branch_0, branch_1, branch_2])
                        residual = slim.conv2d(residual, 256, [1, 1], scope='Conv2d_0a_1x1',
                                    normalizer_fn=None, activation_fn=None)

                shortcut = slim.conv2d(net, 256, [1, 1], stride=1,
                                       normalizer_fn=None, activation_fn=None,
                                       scope='shortcut')

                net = shortcut + residual
                net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='preact')
                end_points[end_point] = net

            net = slim.max_pool2d(net, [3, 3], stride=2, scope='MaxPool_3a_3x3')
            # 输出尺寸 30 x 30 x 256.
            net = cbam_block_channel_first(net, scope='CBAM_Block_Channel_First_3')
            # cbam attention

            with tf.variable_scope('ResBlock2'):
                end_point = 'Conv2d_4a_1x1'
                with slim.arg_scope([slim.conv2d], stride=1, padding='SAME'):
                    with tf.variable_scope('BlockInceptionB'):
                        with tf.variable_scope('Branch_0'):
                            branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                        with tf.variable_scope('Branch_1'):
                            branch_1 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                            branch_1 = tf.concat(axis=3, values=[
                                slim.conv2d(branch_1, 64, [1, 3], scope='Conv2d_0b_1x3'),
                                slim.conv2d(branch_1, 64, [3, 1], scope='Conv2d_0c_3x1')])
                        with tf.variable_scope('Branch_2'):
                            branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                            branch_2 = slim.conv2d(branch_2, 64, [3, 1], scope='Conv2d_0b_3x1')
                            branch_2 = slim.conv2d(branch_2, 64, [1, 3], scope='Conv2d_0c_1x3')
                            branch_2 = tf.concat(axis=3, values=[
                                slim.conv2d(branch_2, 64, [1, 3], scope='Conv2d_0d_1x3'),
                                slim.conv2d(branch_2, 64, [3, 1], scope='Conv2d_0e_3x1')])
                        with tf.variable_scope('Branch_3'):
                            branch_3 = slim.avg_pool2d(net, [3, 3],  stride=1, padding='SAME', scope='AvgPool_0a_3x3')
                            branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                        residual = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
                        residual = slim.conv2d(residual, 512, [1, 1], scope='Conv2d_0a_1x1',
                                               normalizer_fn=None, activation_fn=None)

                shortcut = slim.conv2d(net, 512, [1, 1], stride=1,
                                       normalizer_fn=None, activation_fn=None,
                                       scope='shortcut')

                net = shortcut + residual
                net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='preact')
                end_points[end_point] = net

            # 输出尺寸 30 x 30 x 512.
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='MaxPool_4a_3x3')
            # 输出尺寸 14 x 14 x 512.
            # net = cbam_block_channel_first(net, scope='CBAM_Block_Channel_First_4')
            # cbam attention

            with tf.variable_scope('InceptionBlock3'):
                end_point = 'Conv2d_5a_1x1'
                with slim.arg_scope([slim.conv2d], stride=1, padding='valid'):
                    with tf.variable_scope('BlockInceptionC'):
                        with tf.variable_scope('Branch_0'):
                            branch_0 = slim.conv2d(net, 256, [7, 7], scope='Conv2d_0a_1x1')
                        with tf.variable_scope('Branch_1'):
                            branch_1 = slim.conv2d(net, 512, [1, 1], scope='Conv2d_0a_1x1')
                            branch_1 = slim.conv2d(branch_1, 512, [1, 3], scope='Conv2d_0b_1x3')
                            branch_1 = slim.conv2d(branch_1, 512, [3, 1], scope='Conv2d_0c_3x1')
                            branch_1 = slim.conv2d(branch_1, 512, [5, 5], scope='Conv2d_0e_5x5')
                        with tf.variable_scope('Branch_2'):
                            branch_2 = slim.conv2d(net, 512, [1, 1], scope='Conv2d_0a_1x1')
                            branch_2 = slim.conv2d(branch_2, 512, [7, 1], scope='Conv2d_0b_7x1')
                            branch_2 = slim.conv2d(branch_2, 512, [1, 7], scope='Conv2d_0c_1x7')
                        with tf.variable_scope('Branch_3'):
                            branch_3 = slim.conv2d(net, 256, [1, 1], scope='Conv2d_0a_1x1')
                            branch_3 = slim.conv2d(branch_3, 256, [1, 5], scope='Conv2d_0f_1x5')
                            branch_3 = slim.conv2d(branch_3, 256, [5, 1], scope='Conv2d_0g_5x1')
                            branch_3 = slim.conv2d(branch_3, 256, [3, 3], scope='Conv2d_0h_3x3')
                        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                # 输出尺寸 8 x 8 x 1536.

        return net, end_points
        # Tjnet V2 网络的核心部分，即卷积层部分就完成了


#全局平均池化、Softmax和Auxiliary Logits(之前6e模块的辅助分类节点)
def ihnet(inputs,
          num_classes=6,  # 最后需要分类的数量（比赛数据集的种类数）
          is_training=True,  # 标志是否为训练过程，只有在训练时Batch normalization和Dropout才会启用
          dropout_keep_prob=0.5,  # 节点保留比率
          prediction_fn=tf.sigmoid,  # 最后用来分类的函数
          spatial_squeeze=True,  # 参数标志是否对输出进行squeeze操作（去除维度数为1的维度，比如5*3*1转为5*3）
          reuse=None,  # 是否对网络和Variable进行重复使用
          scope='IhnetV2'):  # 包含函数默认参数的环境

    with tf.variable_scope(scope, 'IhnetV2', [inputs, num_classes],  # 定义参数默认值
                           reuse=reuse) as scope:
        # 'TjnetV2'是命名空间
        with slim.arg_scope([slim.batch_norm, slim.dropout],  # 定义标志默认值
                            is_training=is_training):
            # 拿到最后一层的输出net和重要节点的字典表end_points
            net, end_points = ihnet_base(inputs, scope=scope)  # 用定义好的函数构筑整个网络的卷积部分

            net = slim.avg_pool2d(net, [8, 8], padding='VALID', scope='AvgPool_5a_8x8')
            # 输出尺寸 1 x 1 x 1536.
            net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout1')

            # 处理正常的分类预测逻辑
            # Final pooling and prediction
            # 这一过程的主要步骤：对Mixed_7c的输出进行8*8的全局平均池化>Dropout>1*1*1000的卷积>除去维数为1>softmax分类
            with tf.variable_scope('Logits'):
                # 全连接层 1024
                with tf.variable_scope('Fully'):
                    logits = slim.fully_connected(net, 1024, activation_fn=tf.nn.relu, scope='fc3')
                    logits = slim.dropout(logits, keep_prob=dropout_keep_prob, scope='Dropout')
                # 全连接层 512
                with tf.variable_scope('Fully'):
                    logits = slim.fully_connected(logits, 512, activation_fn=tf.nn.relu, scope='fc2')
                    logits = slim.dropout(logits, keep_prob=dropout_keep_prob, scope='Dropout')

                # 最后的输出
                logits = slim.conv2d(logits, num_classes, [1, 1], activation_fn=None,
                                     normalizer_fn=None, scope='Conv2d_1c_1x1')
                # 激活函数和规范化函数设为空
                #  输出尺寸 1 x 1 x 512.
                if spatial_squeeze:  # tf.squeeze去除输出tensor中维度为1的节点
                    logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')

            end_points['Logits'] = logits
            end_points['Predictions'] = prediction_fn(logits)
            # Softmax对结果进行分类预测

    return end_points['Predictions'], end_points  # 最后返回logits和包含辅助节点的end_points


# end_points里面有'AuxLogits'、'Logits'、'Predictions'分别是辅助分类的输出，主线的输出以及经过softmax后的预测输出

'''
到这里，前向传播已经写完，对其进行运算性能测试
'''

########评估网络每轮计算时间########
def time_tensorflow_run(session, target, info_string):
    # Args:
    # session:the TensorFlow session to run the computation under.
    # target:需要评测的运算算子。
    # info_string:测试名称。

    num_steps_burn_in = 10
    # 先定义预热轮数（头几轮跌代有显存加载、cache命中等问题因此可以跳过，只考量10轮迭代之后的计算时间）
    total_duration = 0.0  # 记录总时间
    total_duration_squared = 0.0  # 总时间平方和  -----用来后面计算方差

    # 迭代计算时间
    for i in range(num_batches + num_steps_burn_in):  # 迭代轮数
        start_time = time.time()  # 记录时间
        _ = session.run(target)  # 每次迭代通过session.run(target)
        duration = time.time() - start_time
        # 每十轮输出一次
        if i >= num_steps_burn_in:
            if not i % 10:
                print('%s: step %d, duration = %.3f' %
                      (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration  # 累加便于后面计算每轮耗时的均值和标准差
            total_duration_squared += duration * duration
    mn = total_duration / num_batches  # 每轮迭代的平均耗时
    vr = total_duration_squared / num_batches - mn * mn
    # 方差，是把一般的方差公式进行化解之后的结果，值得 借鉴
    sd = math.sqrt(vr)  # 标准差
    print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
          (datetime.now(), info_string, num_batches, mn, sd))
    # 输出的时间是处理一批次的平均时间加减标准差


if __name__ == "__main__":
    # 测试前向传播性能
    batch_size = 1  # 因为网络结构较大依然设置为32，以免GPU显存不够
    height, width = 512, 512  # 图片尺寸
    # 随机生成图片数据作为input
    inputs = tf.random_uniform((batch_size, height, width, 3))

    with slim.arg_scope(ihnet_arg_scope()):
        # scope中包含了batch normalization默认参数，激活函数和参数初始化方式的默认值
        logits, end_points = ihnet(inputs, is_training=False)
        # inception_v3中传入inputs获取里logits和end_points

    init = tf.global_variables_initializer()  # 初始化全部模型参数
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    # sess = tf.Session()  # 创建session
    sess.run(init)
    num_batches = 100  # 测试的batch数量
    time_tensorflow_run(sess, logits, "Forward")

    # 分析结果：前面的输出是当前时间下每10步的计算时间，最后输出的是当前时间下前向传播的总批次以及平均时间+-标准差
