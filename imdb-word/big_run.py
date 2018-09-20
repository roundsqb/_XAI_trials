from explain import L2X
from utils import create_dataset_from_score
from validate_explanation import validate
import numpy as np 

k = 10

accs = np.zeros((4,20))

for i in range(20):
	scores, x = L2X(True)
	new_data, new_n_data, new_m_data, new_r_data = create_dataset_from_score(x, scores, k, bigrun = True)
	val_acc_d = validate('some_type', True, new_data)
	val_acc_n = validate('some_type', True, new_n_data)
	val_acc_m = validate('some_type', True, new_m_data)
	val_acc_r = validate('some_type', True, new_r_data)

	accs[:,i] = np.array([val_acc_d, val_acc_n, val_acc_m, val_acc_r])

np.save('data/partial_accs.npy', accs)