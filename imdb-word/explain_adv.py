"""
The code for constructing the original word-CNN is based on
https://github.com/keras-team/keras/blob/master/examples/imdb_cnn.py
"""
   
from keras.layers import Conv1D, Input, GlobalMaxPooling1D, Multiply, Lambda, Embedding, Dense, Dropout, Activation
from keras.datasets import imdb
from keras.engine.topology import Layer 
from keras import backend as K  
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.callbacks import ModelCheckpoint
from keras.models import Model, Sequential 

import numpy as np
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
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from utils import create_dataset_from_score, calculate_acc


# Set parameters:
# Set parameters:
tf.set_random_seed(10086)
np.random.seed(10086)
max_features = 5000
maxlen = 400
batch_size = 40
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 5
k =10 # Number of selected words by L2X.
PART_SIZE = 125
tau=0.5
_EPSILON = K.epsilon()

###########################################
###############Load data###################
###########################################

def load_data():
	"""
	Load data if data have been created.
	Create data otherwise.
	"""

	if 'data' not in os.listdir('.'):
		os.mkdir('data') 
		
	if 'id_to_word.pkl' not in os.listdir('data'):
		print('Loading data...')
		(x_train, y_train), (x_val, y_val) = imdb.load_data(num_words=max_features, index_from=3)
		word_to_id = imdb.get_word_index()
		word_to_id ={k:(v+3) for k,v in word_to_id.items()}
		word_to_id["<PAD>"] = 0
		word_to_id["<START>"] = 1
		word_to_id["<UNK>"] = 2
		id_to_word = {value:key for key,value in word_to_id.items()}

		print(len(x_train), 'train sequences')
		print(len(x_val), 'test sequences')

		print('Pad sequences (samples x time)')
		x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
		x_val = sequence.pad_sequences(x_val, maxlen=maxlen)
		y_train = np.eye(2)[y_train]
		y_val = np.eye(2)[y_val] 

		np.save('./data/x_train.npy', x_train)
		np.save('./data/y_train.npy', y_train)
		np.save('./data/x_val.npy', x_val)
		np.save('./data/y_val.npy', y_val)
		with open('data/id_to_word.pkl','wb') as f:
			pickle.dump(id_to_word, f)	

	else:
		x_train, y_train, x_val, y_val = np.load('data/x_train.npy'),np.load('data/y_train.npy'),np.load('data/x_val.npy'),np.load('data/y_val.npy')
		with open('data/id_to_word.pkl','rb') as f:
			id_to_word = pickle.load(f)

	return x_train, y_train, x_val, y_val, id_to_word

###########################################
###############Original Model##############
###########################################

def create_original_model():
	"""
	Build the original model to be explained. 
	"""
	model = Sequential()
	model.add(Embedding(max_features,
						embedding_dims,
						input_length=maxlen))
	model.add(Dropout(0.2))
	model.add(Conv1D(filters,
					 kernel_size,
					 padding='valid',
					 activation='relu',
					 strides=1))
	model.add(GlobalMaxPooling1D())
	model.add(Dense(hidden_dims))
	model.add(Dropout(0.2))
	model.add(Activation('relu'))
	model.add(Dense(2))
	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy',
				  optimizer='adam',
				  metrics=['accuracy'])

	return model




def generate_original_preds(train = True): 
	"""
	Generate the predictions of the original model on training
	and validation datasets. 
	The original model is also trained if train = True. 
	"""
	x_train, y_train, x_val, y_val, id_to_word = load_data() 
	model = create_original_model()

	if train:
		filepath="models/original.hdf5"
		checkpoint = ModelCheckpoint(filepath, monitor='val_acc', 
			verbose=1, save_best_only=True, mode='max')
		callbacks_list = [checkpoint]
		model.fit(x_train, y_train, validation_data=(x_val, y_val),callbacks = callbacks_list, epochs=epochs, batch_size=batch_size)

	model.load_weights('./models/original.hdf5', 
		by_name=True) 

	pred_train = model.predict(x_train,verbose = 1, batch_size = 1000)
	pred_val = model.predict(x_val,verbose = 1, batch_size = 1000)
	if not train:
		print('The val accuracy is {}'.format(calculate_acc(pred_val,y_val)))
		print('The train accuracy is {}'.format(calculate_acc(pred_train,y_train)))


	np.save('data/pred_train.npy', pred_train)
	np.save('data/pred_val.npy', pred_val) 

###########################################
####################L2X####################
###########################################
# Define various Keras layers.
Mean = Lambda(lambda x: K.sum(x, axis = 1) / float(k), 
	output_shape=lambda x: [x[0],x[2]]) 

class Concatenate(Layer):
	"""
	Layer for concatenation. 
	
	"""
	def __init__(self, **kwargs): 
		super(Concatenate, self).__init__(**kwargs)

	def call(self, inputs):
		input1, input2 = inputs  
		input1 = tf.expand_dims(input1, axis = -2) # [batchsize, 1, input1_dim] 
		dim1 = int(input2.get_shape()[1])
		input1 = tf.tile(input1, [1, dim1, 1])
		return tf.concat([input1, input2], axis = -1)

	def compute_output_shape(self, input_shapes):
		input_shape1, input_shape2 = input_shapes
		input_shape = list(input_shape2)
		input_shape[-1] = int(input_shape[-1]) + int(input_shape1[-1])
		input_shape[-2] = int(input_shape[-2])
		return tuple(input_shape)

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
		logits_ = K.permute_dimensions(logits, (0,2,1))# [batch_size, 1, d]

		d = int(logits_.get_shape()[2])
		unif_shape = [batch_size,self.k,d]

		uniform = K.random_uniform_variable(shape=unif_shape,
			low = np.finfo(tf.float32.as_numpy_dtype).tiny,
			high = 1.0)
		gumbel = - K.log(-K.log(uniform))
		noisy_logits = (gumbel + logits_)/self.tau0
		samples = K.softmax(noisy_logits)
		samples = K.max(samples, axis = 1) 
		logits = tf.reshape(logits,[-1, d]) 
		threshold = tf.expand_dims(tf.nn.top_k(logits, self.k, sorted = True)[0][:,-1], -1)
		discrete_logits = tf.cast(tf.greater_equal(logits,threshold),tf.float32)
		
		#output = K.in_train_phase(samples, discrete_logits) 

		output=samples
		return tf.expand_dims(output,-1)

	def compute_output_shape(self, input_shape):
		return input_shape

def construct_gumbel_selector(X_ph, num_words, embedding_dims, maxlen, tag):
	"""
	Build the L2X model for selecting words. 
	"""
	emb_layer = Embedding(num_words, embedding_dims, input_length = maxlen, name = 'emb_gumbel'+tag)
	emb = emb_layer(X_ph) #(400, 50) 
	net = Dropout(0.2, name = 'dropout_gumbel'+tag)(emb)
	net = emb
	first_layer = Conv1D(100, kernel_size, padding='same', activation='relu', strides=1, name = 'conv1_gumbel'+tag)(net)    

	# global info
	net_new = GlobalMaxPooling1D(name = 'new_global_max_pooling1d_1'+tag)(first_layer)
	global_info = Dense(100, name = 'new_dense_1'+tag, activation='relu')(net_new) 

	# local info
	net = Conv1D(100, 3, padding='same', activation='relu', strides=1, name = 'conv2_gumbel'+tag)(first_layer) 
	local_info = Conv1D(100, 3, padding='same', activation='relu', strides=1, name = 'conv3_gumbel'+tag)(net)  
	combined = Concatenate()([global_info,local_info]) 
	net = Dropout(0.2, name = 'new_dropout_2'+tag)(combined)
	net_out = Conv1D(100, 1, padding='same', activation='relu', strides=1, name = 'conv_last_gumbel'+tag)(net)   
 
	
	return net_out


def _loss_tensor(y_true, y_pred):
	"""
	Loss for the adversarial model. We treat the output as for a binary classification with classes 'correct' and 'not correct',
	and then return -log(p_notcorrect) as the loss to minimise the adversarial selector.

	"""
	y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
	out = K.sum(y_true * y_pred, axis=len(y_pred.get_shape())-1)

	return -K.log(1. - out)

def categorical_inaccuracy(y_true, y_pred):
	"""
	Accuracy measure for the adversarial model: returns 1 if missclassification.

	"""
	return K.mean(K.cast(K.not_equal(K.argmax(y_true, axis=-1),
		K.argmax(y_pred, axis=-1)),
	K.floatx()))


def L2X(train = True): 
	"""
	Generate scores on features on validation by L2X.
	Train the L2X model with variational approaches 
	if train = True. 
	"""
	print('Loading dataset...') 
	x_train, y_train, x_val, y_val, id_to_word = load_data()
	pred_train = np.load('data/pred_train.npy')
	pred_val = np.load('data/pred_val.npy') 
	print('Creating model...')

	train_acc = np.mean(np.argmax(pred_train, axis = 1)==np.argmax(y_train, axis = 1))
	val_acc = np.mean(np.argmax(pred_val, axis = 1)==np.argmax(y_val, axis = 1))
	print('The train and validation accuracy of the original model is {} and {}'.format(train_acc, val_acc))

	# P(S|X) predictor
	with tf.variable_scope('selection_model'):
		X_php = Input(shape=(maxlen,), dtype='int32')
		net_outp = construct_gumbel_selector(X_php, max_features, embedding_dims, maxlen, '_p')
		logits_Tp = Conv1D(1, 1, padding='same', activation=None, strides=1, name = 'conv4_gumbel')(net_outp)
		Tp = Sample_Concrete(tau, k)(logits_Tp)
		emb2p = Embedding(max_features, embedding_dims, 
			input_length=maxlen)(X_php)
		netp = Mean(Multiply()([emb2p, Tp]))

	# P(S|X) anti-predictor
		X_phap = Input(shape=(maxlen,), dtype='int32')
		net_outap = construct_gumbel_selector(X_phap, max_features, embedding_dims, maxlen, '_p')
		logits_Tap = Conv1D(1, 1, padding='same', activation=None, strides=1, name = 'conv4_gumbel')(net_outap)
		Tap = Sample_Concrete(tau, k)(logits_Tap)
		emb2ap = Embedding(max_features, embedding_dims, 
			input_length=maxlen)(X_phap)
		netap = Mean(Multiply()([emb2ap, Tap]))

	# q(X_S)
	with tf.variable_scope('prediction_model'):

		q = Sequential()
		q.add(Dense(hidden_dims))
		q.add(Activation('relu'))
		q.add(Dense(2, activation='softmax', 
			name = 'new_dense'))


	# Train the normal model

	preds = q(netp)
	model=Model(X_php, preds)

	model.compile(loss='categorical_crossentropy',
				  optimizer='rmsprop',#optimizer,
				  metrics=['acc']) 

	if train:
		filepath="models/l2x.hdf5"
		checkpoint = ModelCheckpoint(filepath, monitor='val_acc', 
			verbose=1, save_best_only=True, mode='max')
		callbacks_list = [checkpoint] 
		st = time.time()
		model.fit(x_train, pred_train, 
			validation_data=(x_val, pred_val), 
			callbacks = callbacks_list,
			epochs=epochs, batch_size=batch_size)
		duration = time.time() - st
		print('Training time is {}'.format(duration))		

	model.load_weights('models/l2x.hdf5', by_name=True) 

	#Train the adversary

	q.trainable = False

	anti_preds = q(netap)
	anti_model=Model(X_phap, anti_preds)

	anti_model.compile(loss='categorical_crossentropy',
				  optimizer='rmsprop',#optimizer,
				  metrics=['acc']) 

	new_pred_train = np.zeros(pred_train.shape)
	new_pred_val = np.zeros(pred_val.shape)

	new_pred_val[:,0] = pred_val[:,1]
	new_pred_val[:,1] = pred_val[:,0]

	new_pred_train[:,0] = pred_train[:,1]
	new_pred_train[:,1] = pred_train[:,0]

	if train:
		a_filepath="models/l2xA.hdf5"
		a_checkpoint = ModelCheckpoint(a_filepath, monitor='val_acc', 
			verbose=1, save_best_only=True, mode='max')
		a_callbacks_list = [a_checkpoint] 
		st = time.time()
		anti_model.fit(x_train, new_pred_train, 
			validation_data=(x_val, new_pred_val), 
			callbacks = a_callbacks_list,
			epochs=epochs, batch_size=batch_size)
		duration = time.time() - st
		print('Training time is {}'.format(duration))		

	anti_model.load_weights('models/l2xA.hdf5', by_name=True) 

	pred_model = Model(X_php, logits_Tp) 
	pred_model.compile(loss='categorical_crossentropy', 
		optimizer='adam', metrics=['acc']) 

	scores = pred_model.predict(x_val, 
		verbose = 1, batch_size = batch_size)[:,:,0] 
	scores = np.reshape(scores, [scores.shape[0], maxlen])

	apred_model = Model(X_phap, logits_Tap) 
	apred_model.compile(loss='categorical_crossentropy', 
		optimizer='adam', metrics=['acc']) 

	a_scores = apred_model.predict(x_val, 
		verbose = 1, batch_size = batch_size)[:,:,0] 
	a_scores = np.reshape(a_scores, [a_scores.shape[0], maxlen])

	return scores, x_val, a_scores 

	

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--task', type = str, 
		choices = ['original','L2X'], default = 'original') 
	parser.add_argument('--train', action='store_true')  
	parser.set_defaults(train=False)
	args = parser.parse_args()
	dict_a = vars(args)

	if args.task == 'original':
		generate_original_preds(args.train) 

	elif args.task == 'L2X':
		scores, x, a_scores = L2X(args.train)
		#scores_mean = np.mean(scores,axis=0)
		#new_scores = np.subtract(scores, scores_mean)
		print('Creating dataset with selected words...')
		create_dataset_from_score(x, scores, k, '_')
		np.save('data/scores.npy', scores)
		#np.save('data/new_scores.npy', new_scores)
		print('Creating opposite dataset with selected words..')
		create_dataset_from_score(x, a_scores, k, '_a_')
		np.save('data/a_scores.npy', a_scores)