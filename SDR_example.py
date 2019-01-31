#!/usr/bin/env python3

import sys
import numpy as np
sys.path.append('./src')
from LSDR import *
from sklearn.metrics import accuracy_score


db = {}


db['X'] = np.loadtxt('datasets/breast_30.00.csv', delimiter=',', dtype=np.float64)			
db['Y'] = np.loadtxt('datasets/breast_30.00_label.csv', delimiter=',', dtype=np.int32)			
db['X2'] = np.loadtxt('datasets/breast_30.00_validation.csv', delimiter=',', dtype=np.float64)			
db['Y2'] = np.loadtxt('datasets/breast_30.00_label_validation.csv', delimiter=',', dtype=np.int32)			
db['num_of_clusters'] = 2
db['q'] = 4

sdr = LSDR(db)
sdr.train()

W = sdr.get_projection_matrix()
new_X = sdr.get_reduced_dim_data()

[out_allocation, nmi, svm_time] = use_svm(db['X'], db['Y'], W)
acc_train = accuracy_score(db['Y'], out_allocation)
[out_allocation, nmi_2, svm_time_2] = use_svm(db['X2'], db['Y2'], W)
acc_test = accuracy_score(db['Y2'], out_allocation)

print('Original dimension : %d X %d'%(db['X'].shape[0], db['X'].shape[1]))
print('Reduced dimension : %d X %d'%(new_X.shape[0], new_X.shape[1]))
print('Classification quality in Training Accuracy : %.3f'%(acc_train))
print('Classification quality in Test Accuracy : %.3f'%(acc_test))
print('Classification quality in Training NMI : %.3f'%(nmi))
print('Classification quality in Test NMI : %.3f'%(nmi_2))





