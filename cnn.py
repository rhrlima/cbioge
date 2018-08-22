import sys
import os
import numpy as np 
import pickle
import tensorflow as tf 


image_size = 28
num_labels = 10

beta = 0.01
batch_size = 128
learning_rate = 0.5


def reformat(dataset, labels):
	dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
	labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
	return dataset, labels


def accuracy(predictions, labels):

	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])


def run():
	
	hidden_nodes = 1024

	graph = tf.Graph()

	with graph.as_default():
		
		print('test')

		#Inputs
		tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size*image_size))
		tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
		
		tf_valid_dataset = tf.constant(valid_dataset)
		tf_test_dataset = tf.constant(test_dataset)
		
		#Variables
		weights = {
			'hidden': tf.Variable(tf.truncated_normal([image_size * image_size, hidden_nodes])),
			'output': tf.Variable(tf.truncated_normal([hidden_nodes, num_labels]))
		}
		
		biases = {
			'hidden': tf.Variable(tf.zeros([hidden_nodes])),
			'output': tf.Variable(tf.zeros([num_labels]))
		}
		

		# Model.
		def model(data):
			hidden = tf.matmul(data, weights['hidden']) + biases['hidden']
			hidden = tf.nn.relu(hidden)
			return tf.matmul(hidden, weights['output']) + biases['output']


		logits = model(tf_train_dataset)
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
		
		l2_hidden = tf.nn.l2_loss(weights['hidden'])
		l2_output = tf.nn.l2_loss(weights['output'])
		loss = tf.reduce_mean(loss + beta * (l2_hidden + l2_output))
		
		# Optimizer
		optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
		
		#Prediction for Training, Validation and Test data
		train_prediction = tf.nn.softmax(logits)
		valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
		test_prediction  = tf.nn.softmax(model(tf_test_dataset))
	

	epochs = 3001
	with tf.Session(graph=graph) as session:
		tf.global_variables_initializer().run()
		print("Initialized")
		for step in range(epochs):

			# Pick an offset within the training data, which has been randomized.
			# Note: we could use better randomization across epochs.
			offset = (step * batch_size) % (train_labels.shape[0] - batch_size)

			# Generate a minibatch.
			batch_data = train_dataset[offset:(offset + batch_size), :]
			batch_labels = train_labels[offset:(offset + batch_size), :]

			# Prepare a dictionary telling the session where to feed the minibatch.
			# The key of the dictionary is the placeholder node of the graph to be fed,
			# and the value is the numpy array to feed to it.
			feed_dict = {
				tf_train_dataset : batch_data, 
				tf_train_labels : batch_labels
			}
			_, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
			if (step % 500 == 0):
				print("Minibatch loss at step %d: %f" % (step, l))
				print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
				print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
		print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


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
	run()