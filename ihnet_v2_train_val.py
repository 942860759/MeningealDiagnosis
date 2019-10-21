# -*- coding:utf-8 _*-   
# author: tangwei 
# time: 2018/09/20 11：20
'''
    开始训练
'''
# coding=utf-8

import tensorflow as tf
import numpy as np
import os
from datetime import datetime
import ihnet_v2
import create_tf_record
import tensorflow.contrib.slim as slim
import shutil


labels_nums = 6  # 类别个数 23,77    tongji 0-22岁 国外 0-19岁(3month)
batch_size = 16  #
resize_height = 512  # 指定存储图片高度
resize_width = 512  # 指定存储图片宽度
depths = 3
data_shape = [batch_size, resize_height, resize_width, depths]


# 定义input_images为图片数据
input_images = tf.placeholder(dtype=tf.float32, shape=[None, resize_height, resize_width, depths], name='input')
# 定义input_labels为labels数据
input_labels = tf.placeholder(dtype=tf.float32, shape=[None, labels_nums], name='label')

# 定义dropout的概率
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
is_training = tf.placeholder(tf.bool, name='is_training')


def net_evaluation(sess, loss, accuracy, val_x, val_y, val_nums):
    val_max_steps = int(val_nums / batch_size)
    val_losses = []
    val_accs = []
    for _ in range(val_max_steps):
        # val_x, val_y= sess.run([val_images_batch, val_labels_batch])
        val_loss, val_acc = sess.run([loss, accuracy], feed_dict={input_images: val_x,
                                                                  input_labels: val_y,
                                                                  keep_prob: 1.0,
                                                                  is_training: False})
        val_losses.append(val_loss)
        val_accs.append(val_acc)
    mean_loss = np.array(val_losses, dtype=np.float32).mean()
    mean_acc = np.array(val_accs, dtype=np.float32).mean()
    return mean_loss, mean_acc


def step_train(train_op, loss, accuracy,
               train_images_batch, train_labels_batch, train_nums,  train_log_step,
               val_images_batch, val_labels_batch, val_nums, val_log_step,
               snapshot_prefix, snapshot, end_points):
    '''
    循环迭代训练过程
    :param train_op: 训练op
    :param loss:     loss函数
    :param accuracy: 准确率函数
    :param train_images_batch: 训练images数据
    :param train_labels_batch: 训练labels数据
    :param train_nums:         总训练数据
    :param train_log_step:   训练log显示间隔
    :param val_images_batch: 验证images数据
    :param val_labels_batch: 验证labels数据
    :param val_nums:         总验证数据
    :param val_log_step:     验证log显示间隔
    :param snapshot_prefix: 模型保存的路径
    :param snapshot:        模型保存间隔
    :return: None
    '''
    tf.summary.image('image_origin', end_points['origin_pic'], max_outputs=3)
    # Create summaries to visualize weights
    for var in tf.trainable_variables():
        tf.summary.histogram(var.name, var, family= "weights")

    # Op to calculate every variable gradient
    grads = tf.gradients(loss, tf.trainable_variables())
    grads = list(zip(grads, tf.trainable_variables()))
    # Summarize all gradients
    for grad, var in grads:
        try:
            tf.summary.histogram(var.name + '/gradient', grad, family= "grads")
        except:
            pass
    # Merge all summaries into a single op
    merge_summary = tf.summary.merge_all()
    # merge_summary = tf.summary.merge([image_origin, image_decoder])

    saver = tf.train.Saver(max_to_keep=0)
    # saver = tf.train.Saver()

    # 得到该网络中，所有可以加载的参数
    # variables = tf.contrib.framework.get_variables_to_restore()
    # # 删除output层中的参数
    #
    # variables_to_resotre = [v for v in variables if 'Conv2d_1c_1x1' not in v.name.split('/')]
    # # 构建这部分参数的saver
    # saver = tf.train.Saver(variables_to_resotre)

    max_acc = 0.0
    mean_acc = 0.0
    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.8
    # with tf.Session(config=config) as sess:
    with tf.Session() as sess:
        writer = tf.summary.FileWriter("logs/", sess.graph)  # 第一个参数指定生成文件的目录。

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # 继续训练代码
        ckpt = tf.train.get_checkpoint_state('logs/')
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            try:
                start = int(ckpt.model_checkpoint_path.split('-')[-1])
            except:
                start = int(ckpt.model_checkpoint_path.split('_')[2])
            # saver.restore(sess, 'models_5.0/RSNA/best_models_188500_0.7680.ckpt')
            # start = 188500
        else:
            start = 0
        # 继续训练代码
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        ''' important set here to load dataset cycle'''
        batch_input_images, batch_input_labels = sess.run([train_images_batch, train_labels_batch])
        val_x, val_y = sess.run([val_images_batch, val_labels_batch])
        ''' important set here to load dataset cycle'''

        for i in range(start, max_steps + 1):
            try:
                # batch_input_images, batch_input_labels = sess.run([train_images_batch, train_labels_batch])
                _, train_loss = sess.run([train_op, loss], feed_dict={input_images: batch_input_images,
                                                                      input_labels: batch_input_labels,
                                                                      keep_prob: 0.5, is_training: True})

                # train测试(这里仅测试训练集的一个batch)
                if i % train_log_step == 0:
                    train_acc, train_summary = sess.run([accuracy, merge_summary],
                                                        feed_dict={input_images: batch_input_images,
                                                                   input_labels: batch_input_labels,
                                                                   keep_prob: 1.0,
                                                                   is_training: False
                                                                   })
                    print("%s: Step [%d]  train Loss : %f, training accuracy :  %g" % (
                        datetime.now(), i, train_loss, train_acc))

                    writer.add_summary(train_summary, i)

                    dev_summary = tf.Summary()
                    dev_summary.value.add(tag="train_loss", simple_value=train_loss)
                    dev_summary.value.add(tag="train_acc", simple_value=train_acc)
                    writer.add_summary(dev_summary, i)

                # val测试(测试全部val数据)
                if i % val_log_step == 0:
                    mean_loss, mean_acc = net_evaluation(sess,
                                                         loss,
                                                         accuracy,
                                                         val_x,
                                                         val_y,
                                                         val_nums)
                    print("%s: Step [%d]  val Loss : %f, val accuracy :  %g" % (datetime.now(), i, mean_loss, mean_acc))

                    dev_summary = tf.Summary()
                    dev_summary.value.add(tag="val_loss", simple_value=mean_loss)
                    dev_summary.value.add(tag="val_acc", simple_value=mean_acc)
                    writer.add_summary(dev_summary, i)

                # 模型保存:每迭代snapshot次或者最后一次保存模型
                if (i % snapshot == 0 and i > 0) or i == max_steps or i == start:
                    print('-----save:{}-{}'.format(snapshot_prefix, i))
                    saver.save(sess, snapshot_prefix, global_step=i)

                # 保存val准确率最高的模型
                if mean_acc > max_acc and mean_acc > 0.7:
                    max_acc = mean_acc
                    path = os.path.dirname(snapshot_prefix)
                    best_models = os.path.join(path, 'best_models_{}_{:.4f}.ckpt'.format(i, max_acc))
                    # save_models = os.path.join('models/tongji', 'best_models_{}_{:.4f}.ckpt'.format(i, max_acc))
                    save_models = os.path.join('models', 'best_models_{}_{:.4f}.ckpt'.format(i, max_acc))
                    print('------save:{}'.format(best_models))
                    saver.save(sess, best_models)

                    shutil.copy(best_models + '.data-00000-of-00001', save_models + '.data-00000-of-00001')
                    shutil.copy(best_models + '.index', save_models + '.index')
                    shutil.copy(best_models + '.meta', save_models + '.meta')
            except tf.errors.OutOfRangeError:
                print("%s: Step [%d] read error" % (datetime.now(), i))

        coord.request_stop()
        coord.join(threads)
        writer.close()


def train(train_record_file,
          train_log_step,
          train_param,
          val_record_file,
          val_log_step,
          labels_nums,
          data_shape,
          snapshot,
          snapshot_prefix):
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
    [base_lr, max_steps] = train_param
    [batch_size, resize_height, resize_width, depths] = data_shape

    # 获得训练和测试的样本数
    train_nums = create_tf_record.get_txt_nums(train_record_file)
    val_nums = create_tf_record.get_txt_nums(val_record_file)
    print('train nums:%d,val nums:%d' % (train_nums, val_nums))

    # 从record中读取图片和labels数据,sexs数据
    # train数据,训练数据一般要求打乱顺序shuffle=True
    train_images, train_labels = create_tf_record.read_from_files(train_record_file,
                                                                  resize_height,
                                                                  resize_width,
                                                                  type='normalization')
    train_images_batch, train_labels_batch = create_tf_record.get_batch_images(train_images,
                                                                               train_labels,
                                                                               batch_size=batch_size,
                                                                               shuffle=True)
    # val数据,验证数据可以不需要打乱数据
    val_images, val_labels = create_tf_record.read_from_files(val_record_file,
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

    '''损失函数'''
    class_weights = tf.constant([2., 1., 1., 1., 1., 1.])

    epsilon = 1e-7

    preds = tf.clip_by_value(out, epsilon, 1 - epsilon)
    loss = input_labels * tf.log(preds) + (1.0 - input_labels) * tf.log(1.0 - preds)
    loss_samples = tf.reduce_mean(loss * class_weights, axis=1)

    my_loss = - tf.reduce_mean(loss_samples)
    slim.losses.add_loss(my_loss)

    loss = tf.losses.get_total_loss(add_regularization_losses=True)  # 添加正则化损失loss=2.2
    '''损失函数'''
    accuracy = tf.reduce_mean(tf.cast(tf.abs(out - input_labels), tf.float32) / labels_nums)

    # Specify the optimization scheme:
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=base_lr)
    optimizer = tf.train.AdamOptimizer(learning_rate=base_lr, beta1=0.9, beta2=0.999, epsilon=1e-08)

    # global_step = tf.Variable(0, trainable=False)
    # learning_rate = tf.train.exponential_decay(0.05, global_step, 150, 0.9)

    # 在定义训练的时候, 注意到我们使用了`batch_norm`层时,需要更新每一层的`average`和`variance`参数,
    # 更新的过程不包含在正常的训练过程中, 需要我们去手动像下面这样更新
    # 通过`tf.get_collection`获得所有需要更新的`op`
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # 使用`tensorflow`的控制流, 先执行更新算子, 再执行训练
    with tf.control_dependencies(update_ops):
        # create_train_op that ensures that when we evaluate it to get the loss,
        # the update_ops are done and the gradient updates are computed.
        train_op = slim.learning.create_train_op(total_loss=loss, optimizer=optimizer)

    # 循环迭代过程
    step_train(train_op, loss, accuracy,
               train_images_batch, train_labels_batch, train_nums, train_log_step,
               val_images_batch, val_labels_batch, val_nums, val_log_step,
               snapshot_prefix, snapshot, end_points)


def delete_dire(dire):
    dir_list = []
    for root, dirs, files in os.walk(dire):
        for afile in files:
            os.remove(os.path.join(root, afile))
        for adir in dirs:
            dir_list.append(os.path.join(root, adir))
    for bdir in dir_list:
        os.rmdir(bdir)


def clear_logs():
    log_path = 'logs/'
    ckpt = tf.train.get_checkpoint_state(log_path)
    if ckpt and ckpt.model_checkpoint_path:
        pass
    else:
        delete_dire(log_path)


if __name__ == '__main__':
    train_record_file = 'train_jpg.txt'
    val_record_file = 'val_jpg.txt'

    clear_logs()

    train_log_step = 100  # 显示训练过程log信息间隔
    base_lr = 0.00001  # 学习率
    max_steps = 10000000  # 迭代次数
    train_param = [base_lr, max_steps]

    val_log_step = 500  # 显示验证过程log信息间隔
    snapshot = 500  # 保存文件间隔
    snapshot_prefix = 'logs/model.ckpt'  # 保存模型路径
    train(train_record_file=train_record_file,
          train_log_step=train_log_step,
          train_param=train_param,
          val_record_file=val_record_file,
          val_log_step=val_log_step,
          labels_nums=labels_nums,
          data_shape=data_shape,
          snapshot=snapshot,
          snapshot_prefix=snapshot_prefix)