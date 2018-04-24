import os, pprint, time
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from glob import glob   # 这个是干什么的? 这是返回进行匹配文件,然后返回一个list
from random import shuffle  # 打散数据的
from model import *  # 导入DCGAN模型
from utils import *  # 导入进行数据预处理的相关包

pp = pprint.PrettyPrinter()  # 暂时不用管,这是为了打印美观而准备的

"""
TensorLayer implementation of DCGAN to generate face image.
下面是定义的全局参数,我们是可以在另一个python文件进行引用,命令行参数
Usage : see README.md
Attention: 注意一点这里的flages是全局变量,可以在同级之间进行调用,其中我们在model已经使用部分flags定义的参数
"""
flags = tf.app.flags  # 先定义一个flgs, 使用flags.DEFINE_??可以定义一个命令行参数,第一个参数是参数名字,第二个参数是默认值,第三个参数说明
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")  # 这是整个实验迭代的次数
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")  # Adam learning rate,不是0.001作者在paper中提到了,要不然速度太快
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")  # Adam beta1
flags.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]")  # train_size: inf无限大?为什么需要这个?
flags.DEFINE_integer("batch_size", 64, "The number of batch images [64]")  # batch size
flags.DEFINE_integer("image_size", 108, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_integer("output_size", 64, "The size of the output images to produce [64]")  # output image size
flags.DEFINE_integer("sample_size", 64, "The number of sample images [64]")  # 64张图片构成了一个sample样本
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")  # the output image size
flags.DEFINE_integer("sample_step", 500, "The interval of generating sample. [500]")  # 每隔500次保存一下G网络已经生成的图片
flags.DEFINE_integer("save_step", 500, "The interval of saveing checkpoints. [500]")  # 每隔500保存一下checkpoint
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")  # the download dataset
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")  # 建立checkpoint文件夹存放checkpoint文件
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")  # sample_dir是参数名,sample是建立的文件夹名字
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")  # 是否作为训练集合,主要是使用BN
flags.DEFINE_boolean("is_crop", True, "True for training, False for testing [False]")  # 在64张图片够长的小样本中我们进行crop,也就是进行剪切
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")  # 默认不显示
FLAGS = flags.FLAGS  # 在上面flags定义好之后,我们可以利用flags.FLAGS取出我们放入命令行的参数

def main(_):
    pp.pprint(flags.FLAGS.__flags)  # 打印所以和命令行相关的flag

    tl.files.exists_or_mkdir(FLAGS.checkpoint_dir)  # 如果当前目录系存在checkpoint文件夹则返回True;反之如果当前文件夹不存在checkpoint文件夹则首先建立文件夹然后返回false
    tl.files.exists_or_mkdir(FLAGS.sample_dir)  # 建立samples文件夹这是存放生成图片存放的文件夹

    z_dim = 100  # 这是一开始生成网络输入z分布的维度
    with tf.device("/gpu:1"):  # 假设我们使用GPU 1
        ##========================= DEFINE MODEL ===========================##
        z = tf.placeholder(tf.float32, [FLAGS.batch_size, z_dim], name='z_noise')  # 这是生成网络G默认输入分布的维度
        real_images = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.output_size, FLAGS.output_size, FLAGS.c_dim],
                                     name='real_images')   # 生成网络最后输出图像大小[batch_size, 64, 64, 3] (default)

        # z --> generator for training
        # 第一步我们需要利用G网络将传入的z分布生成假的图像, 训练网络,不进行参数共享
        net_g, g_logits = generator_simplified_api(z, is_train=True, reuse=False)  # net_g是生成网络输出, g_logits是图像输出?
        # generated fake images --> discriminator
        # 第二步我们将第一步生成的假的图像送给D网络进行判别, 训练网络, 不进行参数共享
        net_d, d_logits = discriminator_simplified_api(net_g.outputs, is_train=True, reuse=False)  # net_d是D网络,d_logits是D网络的输出结果,这个是针对
        # real images --> discriminator
        # 第三步我们对真正的图像输入D网络进行判断, 训练网络, 不进行参数共享
        net_d2, d2_logits = discriminator_simplified_api(real_images, is_train=True, reuse=True)  # net_d2 真正图像网络,d2_logits真正图像输出结果
        # sample_z --> generator for evaluation, set is_train to False
        # so that BatchNormLayer behave differently
        # 第四步, 利用上面训练好的G,D网络, 生成我们想要生成的图像. 测试网络不使用BN, 使用参数共享,reuse=True表示使用上面已经训练好的参数模型
        net_g2, g2_logits = generator_simplified_api(z, is_train=False, reuse=True)

        ##========================= DEFINE TRAIN OPS =======================##
        # cost for updating discriminator and generator
        # discriminator: real images are labelled as 1
        d_loss_real = tl.cost.sigmoid_cross_entropy(d2_logits, tf.ones_like(d2_logits), name='dreal')  # 这是是计算真实图像的损失,因为真实数据是1,所以我们使用ones_like产生1数据计算loss
        # discriminator: images from generator (fake) are labelled as 0
        # let `x = logits`, `z = labels`.  The logistic loss is: z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        # 从这里我们可以清楚解决之前的疑惑.第一为什么我们在定义D判断网络需要在网络通过sigmoid函数之前返回一个logits,原因就是我们在利用sigmoid_cross_entropy本身在调用内部函数时候就需要通过sigmoid函数了,所以
        # 我们相当于提前截断他.
        d_loss_fake = tl.cost.sigmoid_cross_entropy(d_logits, tf.zeros_like(d_logits), name='dfake')  # 这里我们对G网络产生的虚假图像计算损失
        d_loss = d_loss_real + d_loss_fake  # D网络的损失判断真实图像的损失加上判断虚假图像损失之和
        # generator: try to make the the fake images look real (1), 也就是从G网路的角度来看,我们想要通过D网络更大可能判断为1,这样G网络损失越小
        g_loss = tl.cost.sigmoid_cross_entropy(d_logits, tf.ones_like(d_logits), name='gfake')

        g_vars = tl.layers.get_variables_with_name('generator', True, True)  # 获取所有的G网络变量的参数,并且打印出来了
        d_vars = tl.layers.get_variables_with_name('discriminator', True, True)  # 获取所以的D网络变量的参数,并且打印出来了
        # 这里的g_vars和d_vars是分别获取G网络和D网络的变量,也是为了下面的ADMM算法服务的
        net_g.print_params(False)  # 打印G网络, 写入log?
        print("---------------")
        net_d.print_params(False)  # 打印D网络,写入log?

        # optimizers for updating discriminator and generator
        d_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) \
                          .minimize(d_loss, var_list=d_vars)
        # 我们可以使用var_list对opt分别进行求解,要不然我们默认是计算所以的`GraphKeys.TRAINABLE_VARIABLES`也就是计算所以的train参数
        # 所以我们可以利用var_list分别对G网络和D网络分别进行训练,而不是一次性对全部网络进行训练,所以这也就是var_list存在的好处
        g_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) \
                          .minimize(g_loss, var_list=g_vars)

    sess = tf.InteractiveSession()  # 使用InteractiveSession仅仅是为了更好的使用交互环境吗?或者简单认为可以使用更少的代码量?
    tl.layers.initialize_global_variables(sess)  # 初始化全局变量

    model_dir = "%s_%s_%s" % (FLAGS.dataset, FLAGS.batch_size, FLAGS.output_size)   # "celebA_64_64"  格式化一个字符串连接
    save_dir = os.path.join(FLAGS.checkpoint_dir, model_dir)   # checkpoint/celebA_64_64 这是save_dir目录
    tl.files.exists_or_mkdir(FLAGS.sample_dir)  # 检查是否存在sample目录,如果没有就创建一个
    tl.files.exists_or_mkdir(save_dir)  # 检查是否存在checkpoint/celebA_64_64文件夹,如果没有就创建一个
    # load the latest checkpoints
    net_g_name = os.path.join(save_dir, 'net_g.npz')  # 现在我们在创建一个子目录, checkpoint/celebA_64_64/net_g.npz
    net_d_name = os.path.join(save_dir, 'net_d.npz')  # checkpoint/celebA_64_64/net_d.npz

    data_files = glob(os.path.join("./data", FLAGS.dataset, "*.jpg"))  # ./表示在最外外面的目录创建一个data文件夹,所以最后是data/celebA/*.jpg
    # glob 是或得当前匹配的内容,是一个list, 那门这么说,data_files里面存放的全面的图片路径, 每一张图片都有一个路径, 最后是一起变成了一个list存放起来了
    sample_seed = np.random.normal(loc=0.0, scale=1.0, size=(FLAGS.sample_size, z_dim)).astype(np.float32)# sample_seed = np.random.uniform(low=-1, high=1, size=(FLAGS.sample_size, z_dim)).astype(np.float32)
    # 产生了z分布~(mean=0, std=1, shape=(64,100)) Gaussian distribution  Q: 为什么这里需要一个sample_size:64
    # 这里的sample_seed相当于是一个测试集,就是正常输入没有通过训练网络
    ##========================= TRAIN MODELS ================================##
    iter_counter = 0
    for epoch in range(FLAGS.epoch):  # 总共迭代25次
        ## shuffle data
        shuffle(data_files)  # 首先把我们得到每一个图片路径进行打散,这样我们shuffle他,就相当于我们现在随机得到了文件路径,间接的打散文件了,要不然我们每次取64张图片还不重复了

        ## update sample files based on shuffled data
        sample_files = data_files[0:FLAGS.sample_size]  # 首先我们将64张图片作为一个样本,这里构成的64张图片路径的List
        sample = [get_image(sample_file, FLAGS.image_size, is_crop=FLAGS.is_crop, resize_w=FLAGS.output_size, is_grayscale = 0) for sample_file in sample_files]  # 列表生成器,每一次取一个路径,然后得到一副图像,然后最后形成64张图像
        # 原图像是image_size:108,为了进行匹配我们在这里进行center_crop将图像中心裁剪成64*64大小
        sample_images = np.array(sample).astype(np.float32)  # 因为get_image返回的是np.array这里的话我们是将列表变成了np.array然后在进行类型转换变成np.float32类型了
        print("[*] Sample images updated!")  # 上面产生的sample_images相当于是我们使用测试样本,作为真实图像输入
        ## load image data
        batch_idxs = min(len(data_files), FLAGS.train_size) // FLAGS.batch_size  # 首先第一点有点不明白为什么来一个inf, 最后结果图像张数总数/batch_size(64) 这里使用批量梯度下降法?
        #  //地板除,结果都是整数
        for idx in range(0, batch_idxs):  # 每次都是取一个batch进行运算,使用batch 梯度下降法
            batch_files = data_files[idx*FLAGS.batch_size:(idx+1)*FLAGS.batch_size]  # 首先每一次取出一个batch图像,这里路径,64为一批
            ## get real images
            # more image augmentation functions in http://tensorlayer.readthedocs.io/en/latest/modules/prepro.html
            batch = [get_image(batch_file, FLAGS.image_size, is_crop=FLAGS.is_crop, resize_w=FLAGS.output_size, is_grayscale = 0) for batch_file in batch_files]  # 中心裁剪成为64,64大小

            batch_images = np.array(batch).astype(np.float32)  # 每一轮都获得64张训练图像,总共多少轮就进行多少次batch,这里得到的真实图像的batch
            batch_z = np.random.normal(loc=0.0, scale=1.0, size=(FLAGS.sample_size, z_dim)).astype(np.float32)  # batch_z = np.random.uniform(low=-1, high=1, size=(FLAGS.batch_size, z_dim)).astype(np.float32)
            # batch_z 得到的由分布得到的batch样本,因为真实样本的是64形成一个batch,那么这里我们z分布也是64个样本,每一个样本是100维
            start_time = time.time()  # 这是进行一轮一个batch开始时间
            # updates the discriminator
            errD, _ = sess.run([d_loss, d_optim], feed_dict={z: batch_z, real_images: batch_images})  # 这里D网络有两个参数,一个是正确图像,另外一个z分布
            # updates the generator, run generator twice to make sure that d_loss does not go to zero (difference from paper)
            for _ in range(2):  # 运行两次G网络,防止出现D网络为0的情况
                errG, _ = sess.run([g_loss, g_optim], feed_dict={z: batch_z})  # 这里的G网络仅仅只有一个输入参数,就是z
            print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                    % (epoch, FLAGS.epoch, idx, batch_idxs, time.time() - start_time, errD, errG))  # 这是运行一个batch所需要的时间,而且记录一下D和G的loss

            iter_counter += 1
            if np.mod(iter_counter, FLAGS.sample_step) == 0:  # 所以的每隔500次保留一下G生成的图片
                # generate and visualize generated images,我们使用没有被训练的数据作为测试集,领用网络学习到的参数生成图像
                img, errD, errG = sess.run([net_g2.outputs, d_loss, g_loss], feed_dict={z : sample_seed, real_images: sample_images})  # 这里的img是归一化之后图像
                tl.visualize.save_images(img, [8, 8], './{}/train_{:02d}_{:04d}.png'.format(FLAGS.sample_dir, epoch, idx))  # 保存的是8*8格式的图像,所以64我们使用sample_size和batch_size都是64就可以理解了
                print("[Sample] d_loss: %.8f, g_loss: %.8f" % (errD, errG))  # 然后每隔500次输出测试集合的误差

            if np.mod(iter_counter, FLAGS.save_step) == 0:  # 每隔500次保存一个网络的参数
                # save current network parameters
                print("[*] Saving checkpoints...")  # 开始保存参数
                tl.files.save_npz(net_g.all_params, name=net_g_name, sess=sess)  # 保存网络G的参数,路径是heckpoint/celebA_64_64/net_g.npz
                tl.files.save_npz(net_d.all_params, name=net_d_name, sess=sess)  # 保存网络D的参数,路径是heckpoint/celebA_64_64/net_d.npz
                print("[*] Saving checkpoints SUCCESS!")  # 保存参数成功

if __name__ == '__main__':
    tf.app.run()  # 脚本,编程范式,因为使用tf.app.flag,所以我们使用tf.app.run()他会自动检测main函数然后运行main
