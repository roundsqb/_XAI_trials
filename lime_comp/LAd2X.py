import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


batch_size = 20
epochs = 5
tau = 0.3
k = 2

def flip_targets(target_vector):
	"""
	Provide the opposite targets for the adversarial loss - 
	this can be played with as much as you want. Note atm it assumes we're passing to x-ent loss

	"""

	if target_vector.max() == 1: # Binary target case
		 return torch.tensor([int(i == 0) for i in target_vector])
	else:
		print('look at flip targets')

def one_hot_embedding(labels, num_classes):
	"""Embedding labels to one-hot form.

	Args:
	  labels: (LongTensor) class labels, sized [N,].
	  num_classes: (int) number of classes.

	Returns:
	  (tensor) encoded labels, sized [N, #classes].
	"""
	y = torch.eye(num_classes) 
	return y[labels] 

def make_selector(input_shape):

	model = nn.Sequential(
		nn.Linear(input_shape, 100),
		nn.ReLU(),
		nn.Dropout(0.2),
		nn.Linear(100,100),
		nn.ReLU(),
		nn.Dropout(0.2),
		nn.Linear(100, input_shape))

	return model

def make_top(input_shape, output_shape):

	model = nn.Sequential(
		nn.Linear(input_shape, 200),
		nn.ReLU(),
		nn.BatchNorm1d(200),
		nn.Linear(200,200),
		nn.ReLU(),
		nn.BatchNorm1d(200),
		nn.Linear(200, output_shape),
		nn.Softmax(dim=1))

	return model

def LAX(data, output_shape, cuda = False, batch_size=batch_size, epochs=epochs, tau=tau, k=k, train = True, reg_weight = 1., valid_size = 0.2, PATHtop = 'data/LAXtop.tar', PATHadv = 'data/LAXadv.tar'):
	"""
	Main LTX wrapper and run. 
	"""
	x_train, pred_train, x_val, pred_val = data
	input_shape = x_train.shape[1] #Assuming data shape is batchxdim
	validation_loss = float('inf')
	n_validation_loss = float('inf')

	top= make_top(input_shape,output_shape)
	adversary = make_selector(input_shape)

	if cuda: #cast to cuda
		adversary.cuda()
		top.cuda()

	train_dataset = torch.utils.data.TensorDataset(x_train, pred_train)
	val_dataset = torch.utils.data.TensorDataset(x_val, pred_val)

	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

	criterion1 = nn.CrossEntropyLoss()
	criterion2 = nn.CrossEntropyLoss() # Replace this with a unique loss and pass to LTX if required. For now, for binary problems, we'll just flip the targets

	optimizer_plus = optim.Adam(top.parameters(), lr = 1e-3) # Updates top of the model.
	optimizer_neg = optim.Adam(adversary.parameters(), lr = 1e-3) # Updates the adversary.

	for epoch in range(epochs):

		running_loss_p = 0.
		val_loss = 0.
		train_correct = 0.
		train_incorrect = 0.

		for i, data in enumerate(train_loader,0):

			inputs, labels = data

			optimizer_plus.zero_grad()

			# Forward pass for normal network
			outputs = top(inputs)
			loss_p = criterion1(outputs, labels)
			
			# Backprop and param step for normal net
			loss_p.backward()
			optimizer_plus.step()

			# print statistics
			running_loss_p += loss_p.item()

			for k in range(labels.shape[0]):
				if outputs[k, labels[k]] > 0.5:
					train_correct += 1.
				else:
					train_incorrect += 1.

			if i % 20 == 19:
				print('[{}, {}] loss: {}'.format(epoch + 1, i + 1, running_loss_p / 20))
				running_loss_p = 0.
			elif i == 0:
				print('[{}, {}] loss: {}'.format(epoch + 1, i + 1, running_loss_p ))

		print('Train accuracy = {}'.format(train_correct / (train_correct + train_incorrect)))

		val_correct = 0.
		val_incorrect = 0.

		for j, data in enumerate(val_loader,0):

			inputs, labels = data

			preds = top(inputs)

			loss = criterion1(preds, labels)

			val_loss += loss.item()

			for k in range(labels.shape[0]):
				if preds[k, labels[k]] > 0.5:
					val_correct += 1.
				else:
					val_incorrect += 1.

			if j % 20 == 19:   
				print('[{}, {}] loss: {}'.format(epoch + 1, i + 1, val_loss / j))

			if j == len(val_loader) - 1 and val_loss / j < validation_loss:
				torch.save({
					'top_state_dict': top.state_dict()
					}, PATHtop)

				validation_loss = val_loss / j

		print('Validation accuracy = {}'.format(val_correct / (val_correct + val_incorrect)))

	best_top = make_top(input_shape,output_shape)
	checkpoint1 = torch.load(PATHtop)
	best_top.load_state_dict(checkpoint1['top_state_dict'])

	for epoch in range(epochs):

		running_loss_n = 0.
		val_loss_n = 0.
		train_correct_n = 0.
		train_incorrect_n = 0.

		for i, data in enumerate(train_loader,0):

			inputs, labels = data

			optimizer_neg.zero_grad()

			# Forward pass for adversarial network
			vecs = adversary(inputs)
			new_inputs = torch.add(vecs, inputs)

			outputs_n = best_top(new_inputs)
			loss_n = criterion2(outputs_n, flip_targets(labels))

			#Regularise vecs to be as small as possible - currently l2
			reg_target = torch.zeros(vecs.shape)
			reg_L = nn.L1Loss()
			#reg_L = nn.MSELoss()
			reg_loss = reg_weight * reg_L(vecs, reg_target)
			
			# Backprop and param step for normal net
			loss_n.backward()
			optimizer_neg.step()

			# print statistics
			running_loss_n += loss_n.item()

			for k in range(labels.shape[0]):
				if outputs_n[k, flip_targets(labels)[k]] > 0.5:
					train_correct_n += 1.
				else:
					train_incorrect_n += 1.

			if i % 20 == 19:
				print('[{}, {}] n_loss: {}'.format(epoch + 1, i + 1, running_loss_n / 20))
				running_loss_n = 0.
			elif i == 0:
				print('[{}, {}] n_loss: {}'.format(epoch + 1, i + 1, running_loss_n ))

		print('Adv train accuracy = {}'.format(train_correct_n / (train_correct_n + train_incorrect_n)))

		val_correct_n = 0.
		val_incorrect_n = 0.

		for j, data in enumerate(val_loader,0):

			inputs, labels = data

			vecs = adversary(inputs)
			new_inputs = torch.add(vecs, inputs)

			preds_n = best_top(new_inputs)

			loss = criterion2(preds_n, flip_targets(labels))

			val_loss_n += loss.item()

			for k in range(labels.shape[0]):
				if preds_n[k, flip_targets(labels)[k]] > 0.5:
					val_correct_n += 1.
				else:
					val_incorrect_n += 1.

			if j % 20 == 19:   
				print('[{}, {}] val n_loss: {}'.format(epoch + 1, j + 1, val_loss_n / j))

			if j == len(val_loader) - 1 and val_loss_n / j < n_validation_loss:
				torch.save({
					'adv_state_dict': adversary.state_dict()
					}, PATHadv)

				n_validation_loss = val_loss_n / j

		print('N-Validation accuracy = {}'.format(val_correct_n / (val_correct_n + val_incorrect_n)))

	scores = torch.zeros((x_val.shape[0],x_val.shape[1]))

	best_adv = make_selector(input_shape)
	checkpoint2 = torch.load(PATHadv)
	best_adv.load_state_dict(checkpoint2['adv_state_dict'])

	best_top.eval()
	best_adv.eval()

	score_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1000, shuffle = False, num_workers=1)

	for k, data in enumerate(score_loader,0):

		inputs, labels = data
		start = k*1000

		scores[start:start+1000,:] = best_adv(inputs)
	

	return scores, best_adv, best_top