import os

print('before import')

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

print('after import')

a = tf.constant("Hello World")
session = tf.Session()
output = session.run(a)

print('end', str(output))
