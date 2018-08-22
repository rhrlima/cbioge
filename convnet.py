from __future__ import print_function

import sys
import os

import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range


def reformat(dataset, labels):
	dataset = dataset.reshape( (-1, image_size, image_size, num_channels) ).astype(np.float32)
	labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
	return dataset, labels


def accuracy(predictions, labels):

	return (100.0 * np.sum( np.argmax(predictions, 1) == np.argmax(labels, 1) ) / predictions.shape[0])


def run():

	graph = tf.Graph()

	with graph.as_default():

		# Input data.
		tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
		tf_train_labels  = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
		tf_valid_dataset = tf.constant(valid_dataset)
		tf_test_dataset  = tf.constant(test_dataset)

		# Variables.
		weights = {
			'layer1': tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1)),
			'layer2': tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1)),
			'layer3': tf.Variable(tf.truncated_normal([image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1)),
			'layer4': tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1)),
		}

		biases = {
			'layer1': tf.Variable(tf.zeros([depth])),
			'layer2': tf.Variable(tf.constant(1.0, shape=[depth])),
			'layer3': tf.Variable(tf.constant(1.0, shape=[num_hidden])),
			'layer4': tf.Variable(tf.constant(1.0, shape=[num_labels])),
		}

		# Model.
		def model(data):
			conv 	= tf.nn.conv2d(data, weights['layer1'], [1, 1, 1, 1], padding='SAME')
			pool 	= tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
			hidden 	= tf.nn.relu(pool + biases['layer1'])

			conv 	= tf.nn.conv2d(hidden, weights['layer2'], [1, 1, 1, 1], padding='SAME') 
			pool 	= tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
			hidden 	= tf.nn.relu(pool + biases['layer2'])

			shape 	= hidden.get_shape().as_list()
			reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
			hidden 	= tf.nn.relu(tf.matmul(reshape, weights['layer3']) + biases['layer3'])
			return tf.matmul(hidden, weights['layer4']) + biases['layer4']

		# Training computation.
		logits = model(tf_train_dataset)
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

		# Optimizer.
		optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

		# Predictions for the training, validation, and test data.
		train_prediction = tf.nn.softmax(logits)
		valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
		test_prediction = tf.nn.softmax(model(tf_test_dataset))


	num_steps = 1001

	with tf.Session(graph=graph) as session:

		tf.global_variables_initializer().run()
		
		print('Initialized')
		for step in range(num_steps):
			offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
			batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
			batch_labels = train_labels[offset:(offset + batch_size), :]
			feed_dict = {
				tf_train_dataset : batch_data,
				tf_train_labels : batch_labels
			}
			_, l, predictions = session.run(
				[optimizer, loss, train_prediction], feed_dict=feed_dict)
			if (step % 50 == 0):
				print('Minibatch loss at step %d: %f' % (step, l))
				print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
				print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
		print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))


if __name__ == '__main__':


	if len(sys.argv) < 2 or not os.path.exists(sys.argv[1]):
		print('No dataset specified, or the file does not exists.', file=sys.stderr)
		exit()


	image_size = 28
	num_labels = 10
	num_channels = 1 # grayscale

	batch_size = 16
	patch_size = 5
	depth = 16
	num_hidden = 64

	pickle_file = sys.argv[1]

	with open(pickle_file, 'rb') as f:
		save = pickle.load(f)

		train_dataset = save['train_dataset']
		train_labels = save['train_labels']

		valid_dataset = save['valid_dataset']
		valid_labels = save['valid_labels']

		test_dataset = save['test_dataset']
		test_labels = save['test_labels']

		del save  # hint to help gc free up memory
		print('Training set', train_dataset.shape, train_labels.shape)
		print('Validation set', valid_dataset.shape, valid_labels.shape)
		print('Test set', test_dataset.shape, test_labels.shape)


	train_dataset, train_labels = reformat(train_dataset, train_labels)
	valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
	test_dataset, test_labels = reformat(test_dataset, test_labels)

	print('Training set', train_dataset.shape, train_labels.shape)
	print('Validation set', valid_dataset.shape, valid_labels.shape)
	print('Test set', test_dataset.shape, test_labels.shape)

	run()
