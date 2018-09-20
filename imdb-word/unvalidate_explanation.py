"""
Compute the accuracy with selected words by using L2X.
"""

import numpy as np
import tensorflow as tf
import os 
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
from explain import create_original_model	

def validate(): 
	print('Loading dataset...')  
	x_val_selected = np.load('data/x_val_n-L2X.npy')
	pred_val = np.load('data/pred_val.npy') 			
	print('Creating model...')
	model = create_original_model()

	weights_name = [i for i in os.listdir('./models') if i.startswith('original')][0]
	model.load_weights('./models/' + weights_name, 
		by_name=True)  
	new_pred_val = model.predict(x_val_selected,verbose = 1, batch_size = 1000)
	val_acc = np.mean(np.argmax(pred_val, axis = -1)==np.argmax(new_pred_val, 
		axis = -1)) 
	print('The validation accuracy with selected 10 bad words is {}.'.format(val_acc)) 

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser() 

	args = parser.parse_args()

	validate()