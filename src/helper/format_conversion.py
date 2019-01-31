#!/usr/bin/env python3

import numpy as np
import torch


def ensure_matrix_is_numpy(U):
	if type(U) == torch.DoubleTensor:
		U = U.numpy()
	elif type(U) == np.ndarray:
		pass
	elif type(U) == torch.FloatTensor:
		U = U.numpy()
	elif type(U) == torch.autograd.variable.Variable:
		U = U.data.numpy()
	else:
		raise
	return U


