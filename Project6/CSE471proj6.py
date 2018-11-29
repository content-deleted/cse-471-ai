import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from load_dataset import mnist 


trainX, trainY, testX, testY = mnist(ntrain=60000,ntest=10000,onehot=False,subset=True,digit_range=[0,10],shuffle=True) # Loading the data


n_input = 28 * 28
n_classes = 10
batch_size = 128
n_epoch = 200
learning_rate = 0.01

features = tf.placeholder(dtype = tf.float32 , shape = [None,n_input])  #placeholder for the input features
labels = tf.placeholder(dtype = tf.float32, shape = [None,n_classes]) #placeholder for the labels

'''
Define your Hidden Layers Here
'''
prev_hidden_layer = None
# Use Relu activation function


'''
-------------------------
'''

logits  = tf.contrib.layers.fully_connected(prev_hidden_layer , n_classes, activation_fn = None)
softmax_op = tf.nn.softmax(logits)
preds = tf.argmax(softmax_op,axis = 1)


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = labels,logits = logits))

correct_prediction = tf.equal(tf.argmax(logits,1) , tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction , tf.float32))


optimizer = tf.train.RMSPropOptimizer(learning_rate)
train_step = optimizer.minimize(loss)

train_losses = []
test_losses = []

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for ii in range(n_epoch):
		for j in range(X.shape[1] // batch_size):
			batch_x = X[batch_size*j : batch_size * j + batch_size, :]
			batch_y = Y_one_hot[batch_size*j : batch_size * j + batch_size, :]
			# Use the batch_x and batch_y to evaluate the train_step tensor to perform the backpropagation.
		if ii % 10 == 0:
			'''
			For Every 10th epoch get the training loss and tesing loss and store them.
			You do it for all the the data points in your training and testing sets, not for batches.
			'''
			train_loss = None
			train_acc = None
			print("Epoch : {}, Training Loss : {}, Training Accuracy : {}").format(ii,train_loss,test_loss)
			train_losses.append(train_loss)
			test_losses.append(train_acc)

test_accuracy = None # Get the test accuracy by evaluating the accuracy tensor with test data and test labels.

'''
YOUR PLOTTING CODE HERE
'''

epochs = range(1, len(test_accuracy) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, train_losses, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, test_losses, 'b', label='Test loss')
plt.title('Training and test loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
