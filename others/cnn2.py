import pickle
import sys
import os

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

IMAGE_SIZE = 28
NUM_LABELS = 10

BETA = 0.01
BATCH_SIZE = 128
LEARNING_RATE = 0.5

HIDDEN_NODES = 1024

STEPS = 2001

def reformat(dataset, labels):
	dataset = dataset.reshape((-1, IMAGE_SIZE * IMAGE_SIZE)).astype(np.float32)
	labels = (np.arange(NUM_LABELS) == labels[:,None]).astype(np.float32)
	return dataset, labels


def accuracy(predictions, labels):

	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])


def run(seed, use_saved_model=False):

	graph = tf.Graph()
	with graph.as_default():

		#Inputs
		tf_train_dataset = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE * IMAGE_SIZE))
		tf_train_labels = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_LABELS))

		tf_valid_dataset = tf.constant(valid_dataset)
		tf_test_dataset = tf.constant(test_dataset)

		#Model
		def model(inputs):
			hidden = tf.contrib.layers.fully_connected(
				inputs=inputs,
				num_outputs=HIDDEN_NODES,
				activation_fn=tf.nn.relu
			)

			logits = tf.contrib.layers.fully_connected(
				inputs=hidden,
				num_outputs=NUM_LABELS,
				activation_fn=tf.nn.relu
			)
			return logits

		logits = model(tf_train_dataset)
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
		
		#l2_loss
		

		#Optimizer
		optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
		
		#Predictions for Training Validation and Test
		train_prediction = tf.nn.softmax(logits)
		valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
		test_prediction = tf.nn.softmax(model(tf_test_dataset))

		#Saver
		saver = tf.train.Saver()


	tf.set_random_seed(seed)

	with tf.Session(graph=graph) as sess:
		tf.global_variables_initializer().run()
		if use_saved_model:
			print('Modelo restaurado')
			saver.restore(sess, '/tmp/model.ckpt')

		for step in range(STEPS):

			offset = (step * BATCH_SIZE) % (train_labels.shape[0] - BATCH_SIZE)

			batch_data = train_dataset[offset:(offset + BATCH_SIZE), :]
			batch_labels = train_labels[offset:(offset + BATCH_SIZE), :]

			feed_dict = {
				tf_train_dataset: batch_data,
				tf_train_labels: batch_labels
			}

			_, l, predictions = sess.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
			if (step % 500 == 0):
				print("Minibatch loss at step %d: %f" % (step, l))
				print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
				print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
		print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

		save_path = saver.save(sess, '/tmp/model.ckpt')
		print('Modelo salvo')


if __name__ == '__main__':
	
	if len(sys.argv) < 2 or not os.path.exists(sys.argv[1]):
		print('No dataset specified, or the file does not exists.', file=sys.stderr)
		exit()

	pickle_file = sys.argv[1]

	print('Loading the dataset')
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


	print('Reshaping')
	train_dataset, train_labels = reformat(train_dataset, train_labels)
	valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
	test_dataset, test_labels = reformat(test_dataset, test_labels)

	print('Training set', train_dataset.shape, train_labels.shape)
	print('Validation set', valid_dataset.shape, valid_labels.shape)
	print('Test set', test_dataset.shape, test_labels.shape)

	print('Running')
	run(42)