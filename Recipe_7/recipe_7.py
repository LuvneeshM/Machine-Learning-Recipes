import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
learn = tf.contrib.learn
tf.logging.set_verbosity(tf.logging.ERROR)

def display(i):
	img = test_data[i]
	plt.title("Example %d. Label: %d" % (i, test_labels[i]))
	plt.imshow(img.reshape((28,28)), cmap=plt.cm.gray_r) #need to reshape since are storing the image in a stacked 1d array (2d array but everything in 1 row)
	plt.show()

print("Loading mnist database")

#for loading the training data (55k images)
mnist = learn.datasets.load_dataset('mnist')
data = mnist.train.images
labels = np.asarray(mnist.train.labels, dtype=np.int32)

#test data (10k images)
test_data = mnist.test.images
test_labels = np.asarray(mnist.test.labels, dtype=np.int32)

max_examples = 10000
data = data[:max_examples]
labels = labels[:max_examples]

#pring some of the test images using display
display(0)