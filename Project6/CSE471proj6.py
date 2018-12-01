import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from load_dataset import mnist 

def one_hot(y,n):
	y_one_hot = np.zeros((y.shape[1],n))
	for i in range(y.shape[1]):
		y_one_hot[i,int(y[0,i])] = 1
	return y_one_hot

def nnArc(layers, activation_function, netNum):
    '''
    Loading the data set.
    Here 
    trainX : trainging data
    trainY : training labels
    testX : testing data
    testY : testing lables
    '''
    trainX, trainY, testX, testY = mnist(ntrain=60000,ntest=10000,onehot=False,subset=True,digit_range=[0,10],shuffle=True) # Loading the data
    trainX = trainX.T
    testX = testX.T


    n_input = 28 * 28 #Number of input neurons
    n_classes = 10 # Number of classes. 
    batch_size = 128
    n_epoch = 200 # Number of iteratios we will perform.
    learning_rate = 0.01

    '''
    As we have multiple classes we first need to perform one hot encoding of our labels
    '''
    train_Y_one_hot = one_hot(trainY, n_classes)
    test_Y_one_hot = one_hot(testY, n_classes)


    features = tf.placeholder(dtype = tf.float32 , shape = [None,n_input])  # placeholder for the input features
    labels = tf.placeholder(dtype = tf.float32, shape = [None,n_classes]) # placeholder for the labels


    '''
    The line below will generate a hidden layers which has features as it's input, 128 neurons and sigmoid as the 
    activation function.
    '''
    hidden_layers = list()

    for i, layer in enumerate(layers):
        hidden_layer = tf.contrib.layers.fully_connected(features if i == 0 else hidden_layers[i-1] , 128 , activation_fn = activation_function)
        hidden_layers.append(hidden_layer)
    
    '''
    Define your Hidden Layers Here. Your last hidden layer should have varianle name "prev_hidden_layer"
    '''
    lay = hidden_layers.pop()

    logits  = tf.contrib.layers.fully_connected(lay , n_classes, activation_fn = None) # Defining the final layer. 

    softmax_op = tf.nn.softmax(logits) # Calculating the softmax activation.
    preds = tf.argmax(softmax_op,axis = 1) # Computing the final predictions.


    '''
    The line below calculates the cross entropy loss for mutliclass predictions.
    '''
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = labels,logits = logits))

    correct_prediction = tf.equal(tf.argmax(logits,1) , tf.argmax(labels,1)) #Comparing network predictons with the actual class labels.
    accuracy = tf.reduce_mean(tf.cast(correct_prediction , tf.float32)) # Computing the accuracy ( How many correct prediction / Total predictions to make)


    optimizer = tf.train.RMSPropOptimizer(learning_rate)

    '''
    This operations does all the important work from calculating the gradients to updating the parameters.
    '''

    train_step = optimizer.minimize(loss)

    train_losses = []
    test_losses = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for ii in range(n_epoch):
            for j in range(trainX.shape[1] // batch_size):
                batch_x = trainX[batch_size*j : batch_size * j + batch_size, :]
                batch_y = train_Y_one_hot[batch_size*j : batch_size * j + batch_size, :]
                sess.run(train_step, feed_dict = {features: batch_x , labels : batch_y})
            if ii % 10 == 0:
                '''
                For Every 10th epoch get the training loss and tesing loss and store them.
                You do it for all the the data points in your training and testing sets, not for batches.
                '''
                train_loss = sess.run(loss,feed_dict = {features : trainX , labels : train_Y_one_hot})
                train_acc = sess.run(accuracy , feed_dict = {features : trainX , labels : train_Y_one_hot})
                test_loss = sess.run(loss , feed_dict = {features : testX , labels : test_Y_one_hot})
                print ('Epoch : {}, Training Loss : {}, Training Accuracy : {}'.format(ii,train_loss,train_acc))
                train_losses.append(train_loss)
                test_losses.append(test_loss)

        test_accuracy = sess.run(accuracy , feed_dict = {features : testX , labels : test_Y_one_hot}) # Get the test accuracy by evaluating the accuracy tensor with test data and test labels.

    '''
    The following code generates the plot for epochs vs training loss and epoch vs testing loss.
    You will need to note the test accuracy and generate a plot for architecture vs test accuracy. 
    '''
    print ('Testing accuracy : {}'.format(test_accuracy))
    
    X_axis = range(1,n_epoch + 1 ,10)
    
    plt.title( 'NN#{} ({})'.format(netNum, activation_function.__name__) )
    plt.plot(X_axis,train_losses,"-",color = "blue")
    plt.plot(X_axis,test_losses,"--",color = "red")
    plt.legend(["Training Loss","Testing Loss"])

    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.show()
    
    return (test_accuracy)

#main 

accuracy = list()

'''
# Data for Number 1
accuracy.append( nnArc(list([2048,1024,512,128,32]), tf.nn.relu, len(accuracy)+1) )
accuracy.append( nnArc(list([1024,512,128,64,32]), tf.nn.relu, len(accuracy)+1) )
accuracy.append( nnArc(list([512,256,128,32,16]), tf.nn.relu, len(accuracy)+1) )
accuracy.append( nnArc(list([256,128,32,16,8]), tf.nn.relu, len(accuracy)+1) )
accuracy.append( nnArc(list([128,64,32,16,8]), tf.nn.relu, len(accuracy)+1) )

'''
'''
# Data for Number 2
accuracy.append( nnArc(list([2048]), tf.nn.relu, len(accuracy)+1) )
accuracy.append( nnArc(list([1024,512]), tf.nn.relu, len(accuracy)+1) )
accuracy.append( nnArc(list([512,256,128]), tf.nn.relu, len(accuracy)+1) )
accuracy.append( nnArc(list([256,128,64,32]), tf.nn.relu, len(accuracy)+1) )
accuracy.append( nnArc(list([256,128,64,32,16]), tf.nn.relu, len(accuracy)+1) )
'''

# Data for Number 3
accuracy.append( nnArc(list([(512,256,128,32,16)]), tf.nn.sigmoid, len(accuracy)+1) )
accuracy.append( nnArc(list([(512,256,128,32,16)]), tf.nn.tanh, len(accuracy)+1) )
accuracy.append( nnArc(list([(512,256,128,32,16)]), tf.nn.relu, len(accuracy)+1) )
accuracy.append( nnArc(list([(512,256,128,32,16)]), tf.nn.leaky_relu, len(accuracy)+1) )
accuracy.append( nnArc(list([(512,256,128,32,16)]), tf.nn.elu, len(accuracy)+1) )


X_axis = range(1, len(accuracy) + 1, 1 )

plt.title('Network Accuracy')

plt.plot(X_axis,accuracy,"bo-")

plt.axis([1,len(accuracy), 0, 1])
plt.xticks(X_axis)
plt.xlabel('NN#')
plt.ylabel('Accuracy')

plt.show()