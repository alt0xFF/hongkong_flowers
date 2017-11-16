import tensorflow as tf
import numpy as np
from keras.applications.imagenet_utils import preprocess_input

def tf_preprocess_input(tf_image):
    r, g, b = tf.split(tf_image, num_or_size_splits=3, axis=3)

    # Zero-center by mean pixel
    b -= 103.939
    g -= 116.779
    r -= 123.68

    # 'RGB'->'BGR'
    data = tf.concat([b, g, r], 3)

    return data

data = np.random.uniform(0, 254, (1, 244, 244, 3))
tf_data = tf.Variable(data, name='tf_data')

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)

    d_tf = sess.run([tf_preprocess_input(tf_data)])[0]
    d = preprocess_input(data)

    print(np.array_equal(d_tf, d))
