import numpy as np
import tensorflow as tf


# Create 100 phony x, y data points in Numpy, y = x * 0.1 + 0.3
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3


X = tf.constant(x_data)
Y = tf.constant(y_data)


W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b


loss = tf.reduce_mean(tf.square(y - Y))
optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

saver = tf.train.Saver()

sess = tf.Session()

saver.restore(sess, '/tmp/model.ckpt')
print('Modelo restaurado')

for step in range(101):
	sess.run(optimizer)
	if step % 20 == 0:
		print(step, sess.run(loss), sess.run(W), sess.run(b))

sess.close()