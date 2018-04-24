"""
这是DCGAN的模型生成部分, G(生成网络,已知分布去卷积产生图像), D(判别网络,对来自Pdata以及Pg的数据进行判断)
网络平衡: Pdata = Pg
几点注意:
1. 作者使用全卷积网络,替代了传统的pooling layer,让网络自己学习采样,我们在判别网络使用了,核心就是将原来的stride=1变成了stride=2
2.  作者移除FCN层,但是没有使用GPL层说是会减小收敛的速度.
3.  使用BN网络,但是没有在所以层都使用,说是会导致样本抖动以及模型的不稳定性,在生成网络最后以及判别网络输入层没有使用BN
4.  生成网络relu + tanh, 判别网络 leaky_relu + sigmoid
这里终于知道为什么生成网络最后一个步骤使用tanh了,因为我们最后归一化网络/125.0 - 1所以生成的范围是[-1, 1]这个刚好是进行归一化结果
另外对于判别网络由于我们是对G网络生成图像以及真实的图像进行判断,D判断真实图像应该越接近1,D判断来自G的图像应该为假所以应该接近0,因此D判断
结果应该是[0,1],所以我们使用sigmoid函数作为最后的输出

"""
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *  # 导入这个layer包我们就可以使用各种层的定义了

flags = tf.app.flags  # 定义flags这样我们就可以取出命令行参数l
FLAGS = flags.FLAGS   # FLAGS作用就是我们可以取出命令行参数

"""
    function: generate a image 
    args:
        inputs: 输入图像
        is_train: 是否处于训练中,主要是为了使用BN;训练阶段使用BN,测试阶段不使用BN
        reuse: 是否进行权值共享; 训练阶段False,测试阶段True
"""
def generator_simplified_api(inputs, is_train=True, reuse=False):
    image_size = 64  # 这个由生成网络最后生成的图像64
    s2, s4, s8, s16 = int(image_size/2), int(image_size/4), int(image_size/8), int(image_size/16)
    gf_dim = 64 # 这个是倒数第二层的滤波器个数, 在原作者的paper中是128, 这里是64
    c_dim = FLAGS.c_dim # 在main.py已经定义了,生成图像颜色数,3,全局参数,可以利用FLAGS取出我们想要的参数结果
    batch_size = FLAGS.batch_size  # 批处理每批的数目64
    w_init = tf.random_normal_initializer(stddev=0.02)  # 各个卷积层参数初始化
    gamma_init = tf.random_normal_initializer(1., 0.02)  # BN算法第四步进行重构,yi = gamma * xi + beta_init, gamma和beta是神经网络需要学习的参数
    with tf.variable_scope("generator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)  # 这个不太重要,已过时函数

        net_in = InputLayer(inputs, name='g/in')  # 这个name命名还是挺有意思的,生成网络是g,判别网络是d
        net_h0 = DenseLayer(net_in, n_units=gf_dim*8*s16*s16, W_init=w_init,
                act = tf.identity, name='g/h0/lin')  # paper中的projection步骤,这里使用FC网络进行维度的扩充, 输出维度维4*4*64*8
        net_h0 = ReshapeLayer(net_h0, shape=[-1, s16, s16, gf_dim*8], name='g/h0/reshape')  # paper中reshape步骤
        net_h0 = BatchNormLayer(net_h0, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g/h0/batch_norm')  # 在送入去卷积层之前我们使用BN网络,为的是减少梯度弥散的问题

        net_h1 = DeConv2d(net_h0, gf_dim*4, (5, 5), out_size=(s8, s8), strides=(2, 2),
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h1/decon2d')  # 这是第一步进行去卷积之后结果,输出结果,[-1, 8,8,256],暂时先不使用任何act,在BN中我们在使用relu
        net_h1 = BatchNormLayer(net_h1, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g/h1/batch_norm')

        net_h2 = DeConv2d(net_h1, gf_dim*2, (5, 5), out_size=(s4, s4), strides=(2, 2),
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h2/decon2d')  # 第二次进行去卷积,输出[-1, 16,16, 128]
        net_h2 = BatchNormLayer(net_h2, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g/h2/batch_norm')  # 作者在paper中强调了在生成网络中我们使用relu,而且在输出层使用tanh,而不是relu

        net_h3 = DeConv2d(net_h2, gf_dim, (5, 5), out_size=(s2, s2), strides=(2, 2),
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h3/decon2d')  # 第三次进行去卷积,输出结果是[-1, 32, 32, 64]
        net_h3 = BatchNormLayer(net_h3, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g/h3/batch_norm')

        net_h4 = DeConv2d(net_h3, c_dim, (5, 5), out_size=(image_size, image_size), strides=(2, 2),
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h4/decon2d')  # 第四次进行去卷积,c_dim是3,输出结果是[-1, 64, 64, 3]
        logits = net_h4.outputs  # 这里的logits是没有整个网络没有进行tanh激活函数输出的结果
        net_h4.outputs = tf.nn.tanh(net_h4.outputs)  # 然后对整个输出网络施加一个tanh的激活函数,然后将结果进行返回
    return net_h4, logits  # net_h4输出的应该是整个网络的信息,而logits输出的是最后生成的图像信息,但是为什么在施加激活函数之前就要输出了?
# 从main.py调用情况我们可以看出, 在z运行G网络之后输出的图像信息,然后我们传入的参数是net_h4.outputs这是经过了tanh函数的,而不是传入logits那么这里的logits仅仅是中间变量吗?


def discriminator_simplified_api(inputs, is_train=True, reuse=False):
    df_dim = 64 # Dimension of discrim filters in first conv layer. [64]
    c_dim = FLAGS.c_dim # n_color 3输入图像[-1, 64, 64, 3]
    batch_size = FLAGS.batch_size # 64
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("discriminator", reuse=reuse):  # 在作者的paper我们知道在判别网络中我们使用leak_relu
        tl.layers.set_name_reuse(reuse)  # 这个201806就会废弃这个函数

        net_in = InputLayer(inputs, name='d/in')
        net_h0 = Conv2d(net_in, df_dim, (5, 5), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2),  # 作者在paper中也提道理leaky_rule的参数是0.2
                padding='SAME', W_init=w_init, name='d/h0/conv2d')  # 因为leak_relu和relu还有一点不一样,他是有参数alpha,我们在调用Conv2d相当于是高阶函数,需要使用lanmda不等式,将act变成一个匿名函数

        net_h1 = Conv2d(net_h0, df_dim*2, (5, 5), (2, 2), act=None,
                padding='SAME', W_init=w_init, name='d/h1/conv2d') # 这里还需要注意一点,我们使用stride=(2,2),而不是(1,1)这样我们经过一个conv层之后图像的大小就会减小一半,而且这里并没有使用maxpooling层,减少了参数的个数
        net_h1 = BatchNormLayer(net_h1, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d/h1/batch_norm') # 这里有一个疑惑为什么不在第一层加入BN?

        net_h2 = Conv2d(net_h1, df_dim*4, (5, 5), (2, 2), act=None,
                padding='SAME', W_init=w_init, name='d/h2/conv2d')  # outputs: [-1, 8, 8, 256]
        net_h2 = BatchNormLayer(net_h2, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d/h2/batch_norm')

        net_h3 = Conv2d(net_h2, df_dim*8, (5, 5), (2, 2), act=None,
                padding='SAME', W_init=w_init, name='d/h3/conv2d') # outputs: [-1, 4, 4, 512]
        net_h3 = BatchNormLayer(net_h3, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d/h3/batch_norm')  # 最后还要经过一次BN网络

        net_h4 = FlattenLayer(net_h3, name='d/h4/flatten')  # 先将tensor变成vector
        net_h4 = DenseLayer(net_h4, n_units=1, act=tf.identity,
                W_init = w_init, name='d/h4/lin_sigmoid')  # 最后输出一个值,是什么,应该是各种网络.
        logits = net_h4.outputs
        net_h4.outputs = tf.nn.sigmoid(net_h4.outputs)  # 最后让网络通过一个sigmoid函数,这是为了之后计算损失函数服务的,因为D网络最后判断真实数据和假的图像
    return net_h4, logits
