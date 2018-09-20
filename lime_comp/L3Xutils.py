import keras
from keras.engine.topology import Layer 
from keras import backend as K  
from keras import regularizers
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.callbacks import ModelCheckpoint
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Input, Multiply, Dropout
from keras.layers.normalization import BatchNormalization
from keras import optimizers
import numpy as np
import pandas as pd
import tensorflow as tf 

batch_size = 40
epochs = 10
tau = 0.2

#################################################
################# Load Data #####################

def load_data(name):

    if name == 'COMPAS_lite':
        data = pd.read_csv('data/propublica_data_for_fairml.csv').as_matrix()
        train_x = data[:4500,1:]
        train_y = data[:4500,0]
        test_x = data[4500:6000,1:]
        test_y = data[4500:6000,0]

        train_y = keras.utils.to_categorical(train_y, 2)
        test_y = keras.utils.to_categorical(test_y, 2)

        return train_x, train_y, test_x, test_y


def get_preds(names):

    if name == 'COMPAS_lite':



def L3X(data, batch_size, epochs=10, tau=0.5, k=2, train = True):

    x_train, pred_train, x_val, pred_val = data 
    input_shape = x_train.shape[1] #Assuming data shape is batchxdim
    output_shape = pred_train.shape[1]
    
    #P(s|X)
    model_input = Input(shape=(input_shape,), dtype='float32')
    net = Dense(100, activation='relu',kernel_regularizer=regularizers.l2(1e-3), name='s/dense1')(model_input)
    net = Dropout(0.2)(net)
    net = Dense(100, activation='relu',kernel_regularizer=regularizers.l2(1e-3), name='s/dense2')(net)
    net = Dropout(0.2)(net)
    logits = Dense(input_shape, name='s/logits')(net)
    samples = Sample_Concrete(tau0=tau,k=k, name='sample')(logits)
    
    #q(X_s)

    new_model_input = Multiply()([model_input, samples])
    #new_model_input = Input(shape=(input_shape,), dtype='float32')
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

    return scores, pred_Model

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

        d = int(logits_.get_shape()[2]) 
        unif_shape = [batch_size,self.k,d] #sizing the sampling.

        uniform = K.random_uniform_variable(shape=unif_shape,
            low = np.finfo(tf.float32.as_numpy_dtype).tiny,
            high = 1.0) #finfo is machine limit for floating precision - this is the draw from a Uniform for Gumbel sftmx
        gumbel = - K.log(-K.log(uniform)) #This is now a tf.tensor; tf.variables are converted to Tensors once used
        noisy_logits = (gumbel + logits_)/self.tau0 
        samples = K.softmax(noisy_logits) #In this context, logits are just 'raw activations'
        samples = K.max(samples, axis = 1) #reduces to max of the softmax (i.e. batch x 784)
        
        logits = tf.reshape(logits,[-1, d]) #Not sure necessary for our dimensions
        threshold = K.expand_dims(tf.nn.top_k(logits, self.k, sorted = True)[0][:,-1], -1) #gives a batchx1 tensor.
                #this is taking the 10th highest logit value as a threshold for each instance
        discrete_logits = K.cast(K.greater_equal(logits,threshold),tf.float32) #Does what you think. Returns a Batch x d vec of zeros or ones 
        
        output = K.in_train_phase(samples, discrete_logits) #Returns samples if in training, discrete_logits otherwise.
        return output #tf.expand_dims(output,-1)

    def compute_output_shape(self, input_shape):
        return input_shape