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

class Sample_Concrete(nn.Module):
	"""
	Gumbel-k layer
	"""

	def __init__(self, D_in, k, tau=0.5, batch_size = batch_size):
		super().__init__()
		self.tau = tau
		self.k = k
		self.batch_size = batch_size

	def forward(self, logits, val=False):
		logits_ = logits.unsqueeze(1) # Want batch x 1 x dim

		#assert logits.size()[0] == batch_size, 'Something gone wrong with dimensions in Sample_Concrete'

		d = int(logits_.size()[2])
		unif_shape = [self.batch_size, self.k, d]

		#Soft sampler for training

		uniform = torch.rand(unif_shape).clamp(min=np.finfo(np.float32).min.item())
		gumbel = -torch.log(-torch.log(uniform))
		noisy_logits = (gumbel + logits_) / self.tau
		samples = F.softmax(noisy_logits, dim=-1)
		samples = torch.max(samples, dim=1)[0]

		#Hard sampling for when not training

		topk, indices = torch.topk(logits, self.k)
		res = torch.zeros(self.batch_size, d)
		discrete_logits = res.scatter(1, indices, 1.)

		if val:
			return discrete_logits
		else:
			return samples

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

def make_variational(input_shape, output_shape):

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


def LTX(data, output_shape, cuda = False, batch_size=batch_size, epochs=epochs, tau=tau, k=k, train = True, reg_weight = 0.01, valid_size = 0.2, PATH = 'data/LTX.tar'):
	"""
	Main LTX wrapper and run. 
	"""
	x_train, pred_train, x_val, pred_val = data
	input_shape = x_train.shape[1] #Assuming data shape is batchxdim
	validation_loss = float('inf')

	# P(s|X), to be trained on the correct loss

	pos_sel = make_selector(input_shape)
	#pos_sc = Sample_Concrete(input_shape, k, tau=tau)
	pos_sc = nn.Tanh()

	# P(s|X), to be trained on the opposite loss (whatever that is)

	neg_sel = make_selector(input_shape)
	#neg_sc = Sample_Concrete(input_shape, k, tau=tau)
	neg_sc = nn.Tanh()

	# Q(Y|X_s) The variational net

	variational = make_variational(input_shape,output_shape)

	if cuda: #cast to cuda
		pos_sel.cuda()
		neg_sel.cuda()
		variational.cuda()

	train_dataset = torch.utils.data.TensorDataset(x_train, pred_train)
	val_dataset = torch.utils.data.TensorDataset(x_val, pred_val)

	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

	criterion1 = nn.CrossEntropyLoss()
	criterion2 = nn.CrossEntropyLoss() # Replace this with a unique loss and pass to LTX if required. For now, for binary problems, we'll just flip the targets

	optimizer_plus = optim.Adam(list(pos_sel.parameters()) + list(variational.parameters()), lr = 1e-3) # Updates both halves of the model.
	optimizer_neg = optim.Adam(neg_sel.parameters(), lr = 1e-3) # Updates the adversarial selector.

	for epoch in range(epochs):

		running_loss_p = 0.
		running_loss_n = 0.
		val_loss = 0.
		train_correct = 0.
		train_incorrect = 0.
		train_correct_n = 0.
		train_incorrect_n = 0.

		for i, data in enumerate(train_loader,0):

			inputs, labels = data

			optimizer_plus.zero_grad()
			optimizer_neg.zero_grad()

			# Forward pass for normal network
			logits = pos_sel(inputs)
			samples = pos_sc(logits)

			new_inputs = inputs * samples

			preds = variational(new_inputs)
			loss_p = criterion1(preds, labels)

			#Forward pass for adversarial network
			logits_n = neg_sel(inputs)
			samples_n = neg_sc(logits_n)

			new_n_inputs = inputs * samples_n

			preds_n = variational(new_n_inputs)
			loss_n = criterion2(preds_n, flip_targets(labels))

			#Jointly regularise logit layers 
			reg_loss = reg_weight * (torch.sum(torch.abs(logits - logits_n)) / logits.shape[0])
			loss_p += reg_loss
			loss_n += reg_loss

			# Backprop and param step for normal net
			loss_p.backward(retain_graph=True)
			optimizer_plus.step()

			# Backprop and param step for adversary
			loss_n.backward()
			optimizer_neg.step()

			# print statistics
			running_loss_p += loss_p.item()
			running_loss_n += loss_n.item()

			for k in range(labels.shape[0]):
				if preds[k, labels[k]] > 0.5:
					train_correct += 1.
				else:
					train_incorrect += 1.

				if preds_n[k, flip_targets(labels)[k]] > 0.5:
					train_correct_n += 1.
				else:
					train_incorrect_n += 1.

			if i % 20 == 19:
				print('[{}, {}] loss: {}, neg_loss: {}'.format(epoch + 1, i + 1, running_loss_p / 20, running_loss_n / 20))
				running_loss_p = 0.
				running_loss_n = 0.
			elif i == 0:
				print('[{}, {}] loss: {}, neg_loss: {}'.format(epoch + 1, i + 1, running_loss_p , running_loss_n))

		print('Train accuracy = {}'.format(train_correct / (train_correct + train_incorrect)))
		print('Adversarial train accuracy = {}'.format(train_correct_n / (train_correct_n + train_incorrect_n)))

		val_correct = 0.
		val_incorrect = 0.
		val_correct_n = 0.
		val_incorrect_n = 0.

		for j, data in enumerate(val_loader,0):

			inputs, labels = data

			logits = pos_sel(inputs)
			#samples = pos_sc(logits, val=True)
			samples = pos_sc(logits)

			new_inputs = inputs * samples

			preds = variational(new_inputs)
			loss = criterion1(preds, labels)

			val_loss += loss.item()

			for k in range(labels.shape[0]):
				if preds[k, labels[k]] > 0.5:
					val_correct += 1.
				else:
					val_incorrect += 1.

				if preds_n[k, flip_targets(labels)[k]] > 0.5:
					val_correct_n += 1.
				else:
					val_incorrect_n += 1.

			if j % 20 == 19:   
				print('[{}, {}] loss: {}'.format(epoch + 1, i + 1, val_loss / j))

			if j == len(val_loader) - 1 and val_loss / j < validation_loss:
				torch.save({
					'p_sel_state_dict': pos_sel.state_dict(), 
					'n_sel_state_dict': neg_sel.state_dict(),
					'var_state_dict': variational.state_dict()
					}, PATH)

				validation_loss = val_loss / j

		print('Validation accuracy = {}'.format(val_correct / (val_correct + val_incorrect)))
		print('Adversarial val accuracy = {}'.format(val_correct_n / (val_correct_n + val_incorrect_n)))

	scores = torch.zeros((x_val.shape[0],x_val.shape[1],2))

	pred_plus = make_selector(input_shape)
	pred_neg = make_selector(input_shape)

	checkpoint = torch.load(PATH)
	pred_plus.load_state_dict(checkpoint['p_sel_state_dict'])
	pred_neg.load_state_dict(checkpoint['n_sel_state_dict'])

	pred_plus.eval()
	pred_neg.eval()

	score_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1000, shuffle = False, num_workers=1)

	for k, data in enumerate(score_loader,0):

		inputs, labels = data
		start = k*1000

		scores[start:start+1000,:,0] = pred_plus(inputs)
		scores[start:start+1000,:,1] = pred_neg(inputs)

	variational.eval()

	return scores, pred_plus, pos_sc, variational, pred_neg, neg_sc



