# coding:utf-8
# 导入tensorflow。
# 这句import tensorflow as tf是导入TensorFlow约定俗成的做法，请大家记住
import tensorflow as tf
# 导入MNIST教学的模块
from tensorflow.examples.tutorials.mnist import input_data
# 与之前一样，读入MNIST数据
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 创建x，x是一个占位符（placeholder），代表待识别的图片
x = tf.placeholder(tf.float32, [None, 784])
# y_是实际的图像标签，同样以占位符表示。
y_ = tf.placeholder(tf.float32, [None, 10])
# W是Softmax模型的参数，将一个784维的输入转换为一个10维的输出
# 在TensorFlow中，变量的参数用tf.Variable表示
W = tf.Variable(tf.zeros([784, 10]))
# b是又一个Softmax模型的参数，我们一般叫做“偏置项”（bias）。
b = tf.Variable(tf.zeros([10]))
# y=softmax(Wx + b)，y表示模型的输出
y = tf.nn.softmax(tf.matmul(x, W) + b)
# 至此，我们得到了两个重要的Tensor：y和y_。
# y是模型的输出，y_是实际的图像标签，不要忘了y_是独热表示的
checkpoint_dir = 'save/'
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
# 创建一个Session
sess = tf.InteractiveSession()
# 创建保存模型的类
saver = tf.train.Saver()
if ckpt and ckpt.model_checkpoint_path:
	saver.restore(sess, ckpt.model_checkpoint_path)
# 正确的预测结果
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# 计算预测准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 计算train data正确率
print('train data accuracy: ', sess.run(accuracy, feed_dict={x: mnist.train.images, y_: mnist.train.labels}))
# 计算验证数据的正确率
print('validation data accuracy:  ', sess.run(accuracy, feed_dict={x: mnist.validation.images, y_: mnist.validation.labels}))
# 计算测试数据的正确率
print('test data accuracy :', sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
