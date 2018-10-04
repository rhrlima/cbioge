import pickle
import sys

import numpy as np
import tensorflow as tf


pickle_file = sys.argv[1]
with open(pickle_file, 'rb') as f:
	temp = pickle.load(f)
	
	train_dataset = temp['train_dataset']
	train_labels = temp['train_labels']

	valid_dataset = temp['valid_dataset']
	valid_labels = temp['valid_labels']

	test_dataset = temp['test_dataset']
	test_labels = temp['test_labels']

	del temp
	print('Training set', train_dataset.shape, train_labels.shape)
	print('Validation set', valid_dataset.shape, valid_labels.shape)
	print('Test set', test_dataset.shape, test_labels.shape)


def reformat(dataset, labels):
	dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
	labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
	return dataset, labels


batch_size = 128
image_size = 28
num_labels = 10

train_dataset, train_labels = reformat(train_dataset, train_labels)


tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size), name='data')
tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))


logits = tf.contrib.layers.fully_connected(
	inputs=tf_train_dataset,
	num_outputs=10,
	activation_fn=tf.nn.relu
)


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

correct_pred = tf.argmax(logits, 1)

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


tf.set_random_seed(1234)
steps = 1001

saver = tf.train.Saver()

with tf.Session() as sess:

	tf.global_variables_initializer().run()
	saver.restore(sess, "/tmp/model.ckpt")
	print('Modelo restaurado')

	for step in range(steps):

		offset = (step * batch_size) % (train_labels.shape[0] - batch_size)

		batch_data = train_dataset[offset:(offset + batch_size), :]
		batch_labels = train_labels[offset:(offset + batch_size), :]

		feed_dict = {
			tf_train_dataset: batch_data,
			tf_train_labels: batch_labels
		}

		_, loss, predictions = sess.run([train_op, accuracy, correct_pred], feed_dict=feed_dict)
		if step % 10 == 0:
			print(step, loss)