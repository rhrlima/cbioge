import tensorflow as tf

op = tf.add(2, 2)

print(op)

with tf.Session() as sess:
	result = sess.run(op)
	print(result)

print('end')