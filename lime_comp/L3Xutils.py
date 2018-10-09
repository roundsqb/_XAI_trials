import keras
from keras.engine.topology import Layer 
from keras import backend as K  
from keras import regularizers
from keras.preprocessing import sequence
from keras.datasets import mnist
from keras.callbacks import ModelCheckpoint
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Input, Multiply, Dropout
from keras.layers.normalization import BatchNormalization
from keras import optimizers
import numpy as np
import pandas as pd
import tensorflow as tf

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

batch_size = 50
epochs = 10
tau = 0.3

#################################################
################# Load Data #####################

def load_data(name):

    if name == 'COMPAS_binary':
        dataframe = pd.read_csv('data/compas_combined.csv')
        labels = dataframe['score_factor'].get_values().astype('float32')
        data = dataframe.drop(columns = ['score_factor', 'decile_score']).get_values().astype('float32')

        train_x = data[:4500,:]
        train_y = labels[:4500]
        test_x = data[4500:6000,:]
        test_y = labels[4500:6000]

        train_y = keras.utils.to_categorical(train_y, 2)
        test_y = keras.utils.to_categorical(test_y, 2)

        return train_x, train_y, test_x, test_y

    elif name == 'COMPAS_ten':
        dataframe = pd.read_csv('data/compas_combined.csv')
        labels = dataframe['decile_score'].get_values().astype('float32')
        data = dataframe.drop(columns = ['score_factor', 'decile_score']).get_values().astype('float32')

        train_x = data[:4500,:]
        train_y = labels[:4500]-1.
        test_x = data[4500:6000,:]
        test_y = labels[4500:6000]-1.

        train_y = keras.utils.to_categorical(train_y, 10)
        test_y = keras.utils.to_categorical(test_y, 10)

        return train_x, train_y, test_x, test_y

    elif name == 'MNIST':
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
############## Get predictions ##################

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
        filepath="models/original_MNIST.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', 
            verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        model.fit(x_train, y_train, validation_data=(x_val, y_val),callbacks = callbacks_list, epochs=epochs, batch_size=batch_size)

    model.load_weights('./models/original_MNIST.hdf5',by_name=True) #If train=False, we assume we have already trained an instance of the model

    return model

def get_preds(name, data):

    train_x, train_y, test_x, test_y = data

    if name == 'COMPAS_binary':
        return train_y, test_y
    elif name == 'COMPAS_ten':
        return train_y, test_y
    elif name == 'MNIST':
        model = create_mnist_model(train=False) #Should have alread trained an MNIST model
        pred_train = model.predict(train_x, verbose=0, batch_size=1000)
        pred_val = model.predict(test_x, vernose=0, batch_size=1000)

        return pred_train, pred_val


#################################################
################# Main Model ####################


def L3X(data, batch_size=batch_size, epochs=epochs, tau=tau, k=3, train = True):

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
        filepath = 'models/L3X.hdf5'
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        model.fit(x_train, pred_train, validation_data=(x_val, pred_val), callbacks=callbacks_list, epochs=epochs, batch_size=batch_size)

    model.load_weights('models/L3X.hdf5', by_name=True)
    
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

#################################################
################## run calls ####################

if __name__ == '__main__':

    import argparse
    parser  = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, choices = ['COMPAS_binary','COMPAS_ten', 'MNIST', 'Train_MNIST'], default='COMPAS_binary')

    args = parser.parse_args()

    if args.task == 'Train_MNIST':
        __ = create_mnist_model()

    else:
        train_x, train_y, test_x, test_y = load_data(args.task)
        pred_train, pred_test = get_preds(args.task, (train_x, train_y, test_x, test_y)) 
        
        scores, pred_Model = L3X((train_x, pred_train, test_x, pred_test))

        np.save('data/scores.npy', scores)
