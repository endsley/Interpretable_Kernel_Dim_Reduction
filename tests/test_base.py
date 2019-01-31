#!/usr/bin/env python3


from sklearn.metrics.cluster import normalized_mutual_info_score
import numpy as np
from subprocess import call
from np_loader import *
from path_tools import *
from ism import *
from orthogonal_optimization import *
from DimGrowth import *
import itertools

#from acc import *
import socket
import torch
import pickle
import random
import string
import os

class test_base():
	def __init__(self, db):
		if db['run_only_validation']:
			db['data_file_name'] = './datasets/' + db['data_name'] + '_validation.csv'
			db['label_file_name'] = './datasets/' + db['data_name'] + '_label_validation.csv'
		else:
			db['data_file_name'] = './datasets/' + db['data_name'] + '.csv'
			db['label_file_name'] = './datasets/' + db['data_name'] + '_label.csv'

		db['validation_data_file_name'] = './datasets/' + db['data_name'] + '_validation.csv'
		db['validation_label_file_name'] = './datasets/' + db['data_name'] + '_label_validation.csv'
		db['best_path'] = '../version9/pre_trained_weights/Best_pk/' 
		db['learning_rate'] = 0.001
		db['center_and_scale'] = True
		db['kernel_type'] = 'rbf'		#rbf, linear, rbf_slow
		db['poly_power'] = 3
		db['poly_constant'] = 1
		self.db = db


		tmp_path = './tmp/' + db['data_name'] + '/'
		db_output_path = tmp_path + 'db_files/'
		batch_output_path = tmp_path + 'batch_outputs/'

		ensure_path_exists('./tmp')
		ensure_path_exists('./results')
		ensure_path_exists(tmp_path)
		ensure_path_exists(db_output_path)
		ensure_path_exists(batch_output_path)




	def remove_tmp_files(self):
		db = self.db
		file_in_tmp = os.listdir('./tmp/' + db['data_name'] + '/db_files/')
		for i in file_in_tmp:
			if i.find(db['data_name']) == 0:
				os.remove('./tmp/' + db['data_name'] + '/db_files/' + i)


	def output_db_to_text(self):
		db = self.db
		db['db_file']  = './tmp/' + db['data_name'] + '/db_files/' + db['data_name'] + '_' +  str(int(10000*np.random.rand())) + '.txt'
		fin = open(db['db_file'], 'w')

		for i,j in db.items():
			if type(j) == str:
				fin.write('db["' + i + '"]="' + str(j) + '"\n')
			elif type(j) == bool:
				fin.write('db["' + i + '"]=' + str(j) + '\n')
			elif type(j) == type:
				fin.write('db["' + i + '"]=' + j.__name__ + '\n')
			elif type(j) == float:
				fin.write('db["' + i + '"]=' + str(j) + '\n')
			elif type(j) == int:
				fin.write('db["' + i + '"]=' + str(j) + '\n')
			elif j is None:
				fin.write('db["' + i + '"]=None\n')
			else:
				raise ValueError('unrecognized type : ' + str(type(j)) + ' found.')

		fin.close()
		return db['db_file']


	def export_bash_file(self, i, test_name, export_db):
		run_name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(2))

		cmd = ''
		cmd += "#!/bin/bash\n"
		cmd += "\n#set a job name  "
		cmd += "\n#SBATCH --job-name=%d_%s_%s"%(i, test_name, run_name)
		cmd += "\n#################  "
		cmd += "\n#a file for job output, you can check job progress"
		cmd += "\n#SBATCH --output=./tmp/%s/batch_outputs/%d_%s_%s.out"%(test_name, i, test_name, run_name)
		cmd += "\n#################"
		cmd += "\n# a file for errors from the job"
		cmd += "\n#SBATCH --error=./tmp/%s/batch_outputs/%d_%s_%s.err"%(test_name, i, test_name, run_name)
		cmd += "\n#################"
		cmd += "\n#time you think you need; default is one day"
		cmd += "\n#in minutes in this case, hh:mm:ss"
		cmd += "\n#SBATCH --time=24:00:00"
		cmd += "\n#################"
		cmd += "\n#number of tasks you are requesting"
		cmd += "\n#SBATCH -N 1"
		cmd += "\n#SBATCH --exclusive"
		cmd += "\n#################"
		cmd += "\n#partition to use"
		cmd += "\n#SBATCH --partition=general"
		cmd += "\n#SBATCH --mem=120Gb"
		cmd += "\n#################"
		cmd += "\n#number of nodes to distribute n tasks across"
		cmd += "\n#################"
		cmd += "\n"
		cmd += "\npython ./src/hsic_algorithms.py " + export_db
		
		fin = open('execute_combined.bash','w')
		fin.write(cmd)
		fin.close()


	def batch_run(self):
		count = 0
		db = self.db
		output_list = self.parameter_ranges()
		every_combination = list(itertools.product(*output_list))

		for count, single_instance in enumerate(every_combination):
			[W_optimize_technique] = single_instance
			db['W_optimize_technique'] = W_optimize_technique
			fname = self.output_db_to_text()
			self.export_bash_file(count, db['data_name'], fname)

			if socket.gethostname().find('login') != -1:
				call(["sbatch", "execute_combined.bash"])
			else:
				os.system("bash ./execute_combined.bash")


	def batch_file_names(self):
		count = 0
		db = self.db
		output_list = self.file_name_ranges()
		every_combination = list(itertools.product(*output_list))

		for count, single_instance in enumerate(every_combination):
			[data_name, W_optimize_technique] = single_instance
			db['data_name'] = data_name
			db['W_optimize_technique'] = W_optimize_technique
			
			tmp_path = './tmp/' + db['data_name'] + '/'
			db_output_path = tmp_path + 'db_files/'
			batch_output_path = tmp_path + 'batch_outputs/'

			ensure_path_exists('./tmp')
			ensure_path_exists(tmp_path)
			ensure_path_exists(db_output_path)
			ensure_path_exists(batch_output_path)

			fname = self.output_db_to_text()
			self.export_bash_file(count, db['data_name'], fname)

			if socket.gethostname().find('login') != -1:
				call(["sbatch", "execute_combined.bash"])
			else:
				os.system("bash ./execute_combined.bash")


	def basic_run(self):
		self.remove_tmp_files()
		fname = self.output_db_to_text()

		call(["./src/hsic_algorithms.py", fname])

