import tensorflow as tf
import numpy as np
from scipy import misc

checkpoint_file = "/home/will/Desktop/Project/saves/model.ckpt"
test_dir = "../Project/test"

imgInput = tf.placeholder(tf.float32, shape=[None, 28, 28, 3])

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape, varName):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial, name=varName)

def bias_variable(shape, varName):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial, name=varName)

W_conv1 = weight_variable([3, 3, 3, 32], "W_conv1")
b_conv1 = bias_variable([32], "b_conv1")

W_conv2 = weight_variable([5, 5, 32, 64], "W_conv2")
b_conv2 = bias_variable([64], "b_conv2")

W_fc1 = weight_variable([7 * 7 * 64, 1024], "W_fc1")
b_fc1 = bias_variable([1024], "b_fc1")

W_fc2 = weight_variable([1024, 2], "W_fc2")
b_fc2 = bias_variable([2], "b_fc2")

h_conv1 = tf.nn.relu(conv2d(imgInput, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

saver = tf.train.Saver()

with tf.Session() as sess:
	saver.restore(sess, checkpoint_file)
	
	test_image = np.array(misc.imread(test_dir + "/other0230.jpeg")).reshape(-1, 28, 28, 3)

	prediction = sess.run(y_conv, feed_dict={imgInput: test_image, keep_prob: 1.0})
	print(prediction)
	print("will = [0, 1], other = [1, 0]")
	print("Model restored.")

