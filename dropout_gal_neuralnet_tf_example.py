#### Uncertainty in Deep Learning
#### To keep the dropout during test time :
#### https://medium.com/towards-data-science/adding-uncertainty-to-deep-learning-ecc2401f2013

#### Another useful link
#### https://github.com/tensorflow/tensorflow/issues/97

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import pdb
import numpy as np



### create TF graph session
tf.reset_default_graph()
sess = tf.Session()

LOGDIR = './graphs'
mnist = input_data.read_data_sets('/tmp/data', one_hot=True)


#defining the model structure
# number of neurons in each hidden layer
HIDDEN1_SIZE = 500
HIDDEN2_SIZE = 250
NUM_CLASSES = 10
NUM_PIXELS = 28 * 28

# experiment with the nubmer of training steps to 
# see the effect
TRAIN_STEPS = 2000
BATCH_SIZE = 100

LEARNING_RATE = 0.01

### creating the inputs for the model
with tf.name_scope('input'):
	# Define inputs
	images = tf.placeholder(dtype=tf.float32, shape=[None, NUM_PIXELS])
	labels = tf.placeholder(dtype=tf.float32, shape=[None, NUM_CLASSES])


# Function to create a fully connected layer
def fc_layer(input, size_out, name="fc", activation=None):
    with tf.name_scope(name):
        size_in = int(input.shape[1])
        w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="weights")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="bias")

        #output of the network
        wx_plus_b = tf.matmul(input, w) + b

        if activation: return activation(wx_plus_b)
        return wx_plus_b


### defining the network here
fc1 = fc_layer(images, HIDDEN1_SIZE, "fc1", activation=tf.nn.relu)
fc2 = fc_layer(fc1, HIDDEN2_SIZE, "fc2", activation=tf.nn.relu)
dropped = tf.nn.dropout(fc2, keep_prob=0.9)
#### network output
y = fc_layer(dropped, NUM_CLASSES, name="output")



with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=labels))
    tf.summary.scalar('loss', loss)


with tf.name_scope("optimizer"):
    train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

# Define evaluation
with tf.name_scope("evaluation"):
	prediction = tf.argmax(y, 1)

	correct_prediction = tf.equal(prediction, tf.argmax(labels, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	tf.summary.scalar('accuracy', accuracy)


train_writer = tf.summary.FileWriter(os.path.join(LOGDIR, "train"))
train_writer.add_graph(sess.graph)
test_writer = tf.summary.FileWriter(os.path.join(LOGDIR, "test"))
summary_op = tf.summary.merge_all()

##### Constructing TF graph upto this


sess.run(tf.global_variables_initializer())


MC_SAMPLES = 20
for step in range(TRAIN_STEPS):
    batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)

    ### training results
    summary_result, _ = sess.run([summary_op, train], feed_dict={images: batch_xs, labels: batch_ys})


    orig_predicted_y = sess.run([y], feed_dict = {images : mnist.test.images})
    

    ## use a placeholder here?
    #All_MC_Predicted_Classes = tf.placeholder(dtype=tf.float32, shape=[10000, MC_SAMPLES])    
    All_MC_Predicted_Classes = np.zeros(shape=(10000, MC_SAMPLES))


    for m in range(MC_SAMPLES):
    	predicted_y = sess.run([y], feed_dict = {images : mnist.test.images})

        ### using numpy`
    	predicted_y = np.asarray(predicted_y)
    	predicted_y = predicted_y[0, :, :]
    	predicted_class = np.argmax(predicted_y, 1)
    	pred = np.array([predicted_class]).T
    	All_MC_Predicted_Classes[:, m] = predicted_class


        #predicted_class = tf.argmax(predicted_y, 1)
    	# All_MC_Predicted_Classes[:, m] = tf.argmax(predicted_y, 1)

	Mean_Predicted_Classes = np.mean(All_MC_Predicted_Classes, axis=1)
    	Variance_Predicted_Classes = np.var(All_MC_Predicted_Classes, axis=1)
	# Mean_Predicted_Classes = tf.reduce_mean(All_MC_Predicted_Classes, axis=1) 	
	# Mean_Predicted_Classes = Mean_Predicted_Classes.tolist()


    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    #acc = sess.run(accuracy, feed_dict={images: mnist.test.images, labels: mnist.test.labels})


    print ("Step", step)
    # print ("Test Accuracy", acc)










"""
Doing computation on the TF graph
"""

# sess.run(tf.global_variables_initializer())
# ###### Example to calculate MC Sample based accuracy using MCDropout
# MC_SAMPLES = 10
# for step in range(TRAIN_STEPS):
#     batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
#     summary_result, train_result = sess.run([summary_op, train], feed_dict={images: batch_xs, labels: batch_ys})

#     # calculate accuracy on the test set, every 100 steps.
#     # we're using the entire test set here, so this will be a bit slow
#     if step % 100 == 0:
#     	print ("Step Number", step)
#         #_, acc = sess.run([summary_op, accuracy], feed_dict={images: mnist.test.images, labels: mnist.test.labels})

#         All_Accuracy = np.zeros(shape=([MC_SAMPLES]))
#         for m in range(MC_SAMPLES):
# 			_, acc = sess.run([summary_op, accuracy], feed_dict={images: mnist.test.images, labels: mnist.test.labels})
# 			acc = np.array([acc])

# 			All_Accuracy[m] = acc

# 	print ("All Accuracy", All_Accuracy)
# 	print ("Mean Accuracy", np.mean(All_Accuracy))







"""
To keep the dropout at test time
"""

# keep the dropout during test time
#mc_post = [sess.run(nn, feed_dict={x: data}) for _ in range(100)]

#and then we need sample variance + inverse precision
# def _tau_inv(keep_prob, N, l2=0.005, lambda_=0.00001):
#     tau = keep_prob * l2 / (2. * N * lambda_)
#     return 1. / tau

# np.var(mc_post) + _tau_inv(0.5, 100)
