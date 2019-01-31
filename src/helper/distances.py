

import numpy as np
from format_conversion import *
import sklearn.metrics


def median_of_pairwise_distance(U):
	U = ensure_matrix_is_numpy(U)
	if np.sum(np.std(U,axis=0)) < 0.00001:	#	This condition is triggered when every sample is identical
		print('Error : every sample is now identical')
		import pdb; pdb.set_trace()

	return np.median(sklearn.metrics.pairwise.pairwise_distances(U))


