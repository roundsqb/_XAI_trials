import pandas as pd 
import numpy as np 
import pickle as pickle
import os 
import csv

def get_selected_words(x_single, score, id_to_word, k): 
	selected_words = {} # {location: word_id}

	selected = np.argsort(score)[-k:] 
	selected_k_hot = np.zeros(400)
	selected_k_hot[selected] = 1.0

	x_selected = (x_single * selected_k_hot).astype(int)
	return x_selected 

def get_negative_words(x_single, score, id_to_word, k):

	Nselected = np.argsort(score)[:k]
	Nselected_k_hot = np.zeros(400)
	Nselected_k_hot[Nselected] = 1.0

	Nx_selected = (x_single * Nselected_k_hot).astype(int)
	return Nx_selected

def get_middle_words(x_single, score, id_to_word, k):

	Mselected = np.argsort(score)[int(190-k/2):int(190+k/2)]
	Mselected_k_hot = np.zeros(400)
	Mselected_k_hot[Mselected] = 1.0

	Mx_selected = (x_single * Mselected_k_hot).astype(int)
	return Mx_selected

def get_random_words(x_single, score, id_to_word, k):

	Rselected = np.random.choice(range(400), size=k, replace=False)
	Rselected_k_hot = np.zeros(400)
	Rselected_k_hot[Rselected] = 1.0

	Rx_selected = (x_single * Rselected_k_hot).astype(int)
	return Rx_selected

def create_dataset_from_score(x, scores, k, tag, bigrun = False):
	with open('data/id_to_word.pkl','rb') as f:
		id_to_word = pickle.load(f)
	new_data = []
	new_n_data = []
	new_m_data = []
	new_r_data = []

	for i, x_single in enumerate(x):
		x_selected = get_selected_words(x_single, 
			scores[i], id_to_word, k)
		x_n_selected = get_negative_words(x_single,
			scores[i], id_to_word, k)
		x_m_selected = get_middle_words(x_single,
			scores[i], id_to_word, k)
		x_r_selected = get_random_words(x_single,
			scores[i], id_to_word, k)

		new_data.append(x_selected) 
		new_n_data.append(x_n_selected)
		new_m_data.append(x_m_selected)
		new_r_data.append(x_r_selected)

	if bigrun:
		return new_data, new_n_data, new_m_data, new_r_data

	else:
		np.save('data/x_val-L2X'+tag+'n.npy', np.array(new_data))
		np.save('data/x_val_n-L2X'+tag+'n.npy', np.array(new_n_data))
		np.save('data/x_val_m-L2X'+tag+'n.npy', np.array(new_m_data))
		np.save('data/x_val_r-L2X'+tag+'n.npy', np.array(new_r_data))

def calculate_acc(pred, y):
	return np.mean(np.argmax(pred, axis = 1) == np.argmax(y, axis = 1))
	
	


