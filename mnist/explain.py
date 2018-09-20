
import numpy as np
from numpy import nan
import tensorflow as tf 
import time 
import numpy as np 
import sys
import os
import urllib.request, urllib.error, urllib.parse 
import tarfile
import zipfile 
try:
	import pickle as pickle
except:
	import pickle
import os
import keras

os.environ["CUDA_VISIBLE_DEVICES"]="1"

from keras.datasets import mnist
from keras.engine.topology import Layer 
from keras import backend as K  
from keras import regularizers
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Input, Multiply
from keras.layers.normalization import BatchNormalization
from keras import optimizers

batch_size = 40
num_classes = 10
epochs = 10
tau = 0.2
k = 28

#################################################
################# Load Data #####################

def load_data():
	"""
	Load Data from keras mnist dataset, adjust to appropriate dimensions range etc.
	"""
	(x_train, y_train), (x_val, y_val) = mnist.load_data()
	x_train = x_train.reshape(60000, 784)
	x_val = x_val.reshape(10000, 784)
	x_train = x_train.astype('float32')
	x_val = x_val.astype('float32')
	x_train /= 255
	x_val /= 255
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_val = keras.utils.to_categorical(y_val, num_classes)

	return x_train, y_train, x_val, y_val

#################################################
################## Models #######################

def create_mnist_model(train = True):
	"""
	Build simple MNIST model in Keras, and train it if train = True
	"""

	model = Sequential()
	model.add(Dense(100, activation='relu', input_shape=(784,)))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(25, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))

	model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

	x_train, y_train, x_val, y_val = load_data()

	if train:
		filepath="models/original.hdf5"
		checkpoint = ModelCheckpoint(filepath, monitor='val_acc', 
			verbose=1, save_best_only=True, mode='max')
		callbacks_list = [checkpoint]
		model.fit(x_train, y_train, validation_data=(x_val, y_val),callbacks = callbacks_list, epochs=epochs, batch_size=batch_size)

	model.load_weights('./models/original.hdf5',by_name=True) #If train=False, we assume we have already trained an instance of the model

	return model

def generate_model_preds(model, x_train, x_val):
	"""
	Given a model and input data, save the model's predictions
	"""

	pred_train = model.predict(x_train,verbose=1, batch_size=1000)
	pred_val = model.predict(x_val,verbose=1,batch_size=1000)

	np.save('data/pred_train.npy', pred_train)
	np.save('data/pred_val.npy', pred_val)

	return pred_train, pred_val

#################################################
################## L2X ##########################

class Sample_Concrete(Layer):
    """
    Layer for sample Concrete / Gumbel-Softmax variables. 

    """
    def __init__(self, tau0, k, **kwargs): 
        self.tau0 = tau0
        self.k = k
        super(Sample_Concrete, self).__init__(**kwargs)

    def call(self, logits):   
        # logits: [batch_size, d, 1]
        logits_ = K.expand_dims(logits, -2) #transform to Batch x 1 x Dim

        d = int(logits_.get_shape()[2]) #d = 784 in this case
        unif_shape = [batch_size,self.k,d] #sizing the sampling.

        uniform = K.random_uniform_variable(shape=unif_shape,
            low = np.finfo(tf.float32.as_numpy_dtype).tiny,
            high = 1.0) #finfo is machine limit for floating precision - this is the draw from a Uniform for Gumbel sftmx
        gumbel = - K.log(-K.log(uniform)) #This is now a tf.tensor; tf.variables are converted to Tensors once used
        noisy_logits = (gumbel + logits_)/self.tau0 
        samples = K.softmax(noisy_logits) #In this context, logits are just 'raw activations'
        samples = K.max(samples, axis = 1) #reduces to max of the softmax (i.e. batch x 784)
        
        logits = tf.reshape(logits,[-1, d]) #Not sure necessary for our dimensions
        threshold = tf.expand_dims(tf.nn.top_k(logits, self.k, sorted = True)[0][:,-1], -1) #gives a batchx1 tensor.
                #this is taking the 10th highest logit value as a threshold for each instance
        discrete_logits = tf.cast(tf.greater_equal(logits,threshold),tf.float32) #Does what you think. Returns a Batch x d vec of zeros or ones 
        
        output = K.in_train_phase(samples, discrete_logits) #Returns samples if in training, discrete_logits otherwise.
        return output #tf.expand_dims(output,-1)

    def compute_output_shape(self, input_shape):
        return input_shape

def L2X(Bigmodel, data, train = True):

	x_train, y_train, x_val, y_val = data
	input_shape = x_train.shape[1] #Assuming data shape is batchxdim
	output_shape = y_train.shape[1] #Classification should be 1-hot

	pred_train, pred_val = generate_model_preds(Bigmodel, x_train, x_val)

	#P(s|X)
	model_input = Input(shape=(input_shape,), dtype='float32')
	net = Dense(100, activation='relu',kernel_regularizer=regularizers.l2(1e-3), name='s/dense1')(model_input)
	net_A = Dense(100, activation='relu',kernel_regularizer=regularizers.l2(1e-3), name='s/dense2')(net)
	logits = Dense(input_shape, name='s/logits')(net_A)
	samples = Sample_Concrete(tau0=tau,k=28, name='sample')(logits)

	#q(X_s)

	new_model_input = Multiply()([model_input, samples])
	net = Dense(200, activation='relu', kernel_regularizer=regularizers.l2(1e-3), name='dense1')(new_model_input)
	net = BatchNormalization()(net)
	net = Dense(200, activation='relu', kernel_regularizer=regularizers.l2(1e-3), name='dense2')(net)
	net = BatchNormalization()(net)
	preds = Dense(output_shape, activation='softmax',kernel_regularizer=regularizers.l2(1e-3), name='dense3')(net)

	model=Model(model_input,preds)

	if train:
		adam = optimizers.Adam(lr = 1e-3)
		model.compile(loss = 'categorical_crossentropy',
			optimizer=adam,
			metrics=['acc'])
		filepath = 'models/L2X.hdf5'
		checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
		callbacks_list = [checkpoint]
		model.fit(x_train, pred_train, validation_data=(x_val, pred_val), callbacks=callbacks_list, epochs=epochs, batch_size=batch_size)

	else:
		model.load_weights('models/L2X.hdf5', by_name=True)

	pred_Model = Model(model_input,logits)
	pred_Model.compile(loss=None,
		optimizer='adam',
		metrics=[None])

	scores = pred_Model.predict(x_val, verbose=1, batch_size=batch_size)

	#Get grads of loss wrt learned weighting

	weights = model.trainable_weights
	gradients = model.optimizer.get_gradients(model.total_loss, weights)

	input_tensors = [model.inputs[0], model.sample_weights[0], model.targets[0], K.learning_phase()]

	get_gradients = K.function(inputs=input_tensors, outputs=gradients)

	dC_dLogits = np.zeros(x_val.shape)

	activation_pred = Model(model_input, net_A)
	activation_pred.compile(loss = 'categorical_crossentropy',
			optimizer=adam,
			metrics=['acc'])
	activations = activation_pred.predict(x_val, verbose=1, batch_size=batch_size)

	for j in range(x_val.shape[0]):
		if j % 1000 == 0:
			print('{}/{} grads extracted'.format(j, x_val.shape[0]))
		inputs = [x_val[j:j+1,:], [1.], y_val[j:j+1,:], 1] #Think we want train mode
		grds = get_gradients(inputs)
		deltas = np.divide(grds[4], np.swapaxes(activations[j:j+1,:], 0, 1))

		delts = np.zeros((deltas.shape[1]))

		if j % 1000 == 0:
			print(deltas)

		for i in range(deltas.shape[0]):
			if not np.isnan(deltas[i,0]):
				delts = deltas[i,:]
				break

		dC_dLogits[j,:] = delts

	return scores, dC_dLogits

def validate_L2X(n_feats, trained_model, x_val, pred_val, scores):
	top_n = n_feats

	truncated_val = np.zeros(x_val.shape)

	for i in range(x_val.shape[0]):
		selected = np.argsort(scores[i,:])[-top_n::]
		#selected = np.random.choice(range(784), size=top_n, replace=False)
		selected_k_hot = np.zeros(784)
		selected_k_hot[selected] = 1.0
		truncated_val[i,:] = np.multiply(x_val[i,:],selected_k_hot)

	new_pred_val = trained_model.predict(truncated_val, verbose=0, batch_size=1000)
	val_acc = np.mean(np.argmax(pred_val, axis = -1)==np.argmax(new_pred_val, 
		axis = -1)) 
	#print('The validation accuracy with {} pixels is {}.'.format(top_n, val_acc))

	return val_acc


if __name__ == '__main__':

	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--task', type=str, choices = ['trainMNIST','L2X', 'recovery', 'grad_recovery'], default='trainMNIST')
	parser.add_argument('--train', action='store_true')
	parser.set_defaults(train=False)

	args = parser.parse_args()

	if args.task == 'trainMNIST':
		model = create_mnist_model(args.train)
	elif args.task == 'L2X':

		model = create_mnist_model(train=False)

		x_train, y_train, x_val, y_val = load_data()

		scores, grads = L2X(model, [x_train, y_train, x_val, y_val], args.train)

		np.save('data/scores.npy', scores)
		np.save('data/grads.npy', grads)

	elif args.task == 'recovery':
		trained_model = keras.models.load_model('models/original.hdf5')
		accs = np.zeros(784)

		x_train, y_train, x_val, y_val = load_data()
		pred_val = np.load('data/pred_val.npy')
		scores = np.load('data/scores.npy')

		for i in range(len(accs)):
			accs[i] = validate_L2X(i+1, trained_model, x_val, pred_val, scores)
		np.save('data/rec_accs_r.npy', accs)

	elif args.task == 'grad_recovery':
		trained_model = keras.models.load_model('models/original.hdf5')
		accs = np.zeros(784)

		x_train, y_train, x_val, y_val = load_data()
		pred_val = np.load('data/pred_val.npy')
		scores = np.load('data/grads.npy')

		for i in range(len(accs)):
			if i % 28 == 0:
				print('{}/{} done'.format(i, len(accs)))
			accs[i] = validate_L2X(i+1, trained_model, x_val, pred_val, scores)
		np.save('data/grad_accs_n.npy', accs)


