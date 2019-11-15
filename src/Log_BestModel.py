import datetime
import pandas as pd
import numpy as np
import ast
import logging

path1 = '../'
path = '../BestModel/'

def setup_logger(name, log_file, level=logging.INFO):
	formatter = logging.Formatter('%(asctime)s [%(filename)s: %(funcName)s - %(lineno)d] - %(message)s', datefmt='%d-%b-%y %H:%M:%S)')
	handler = logging.FileHandler(log_file)
	handler.setFormatter(formatter)
	logger = logging.getLogger(name)
	logger.setLevel(level)
	logger.addHandler(handler)
	return logger


# Custom logging class - contains read and save function for each classifier
# Saving model
class Log_BestModel:

	# filename
	# Model 1
	DecTree_filename = path+"Decision_Tree.txt"
	# Model 2
	KNN_filename = path+"KNN.txt"
	# Model 3
	LogReg_filename = path+"Logistic_Regression.txt"
	# Model 4
	RanFo_filename = path+"Random_Forest.txt"
	# Model 5
	SVM_filename = path+"Support_Vector_Machine.txt"

	logger = None


	def __init__(self):
		try:
			logger = setup_logger('LogBestModel', path1 + 'Logging/LogBestModel.log')
			self.logger = logger
			self.logger.info('class Initialization')
			print("------------------------------ LogBestModel Initialized  ---------------------")
		except Exception as e:
			logging.info('Error in __init__')
			logging.error("Exception occurred in __init__", exc_info=True)
			raise e
		return None

	# save best parameters in custom logging file - this function is usually called during the hyperparameter optimization
	def save_Random_Forest_BestModel(self,parammeters):
		try:
			now = datetime.datetime.now()
			currentDate =",CurrentDate:"+now.strftime("%Y-%m-%d %H-%M")+"\n"

			parammeters=parammeters+currentDate
			f = open(self.RanFo_filename, "a+")
			f.write(parammeters)
			f.close()
		except Exception as e:
			logging.info('Error in save_Random_Forest_BestModel')
			logging.error("Exception occurred in save_Random_Forest_BestModel", exc_info=True)
			raise e
		return None

	# read best parameters from custom logging file
	def read_Random_Forest_BestModel(self):
		try:
			BestModel_list = {}
			i = 1
			with open(self.RanFo_filename) as f:
				for line in f:
					hyperparameters = {
						'accuracy_score': None,
						'n_estimators': None,
						'criterion': None,
						'max_depth': None,
						'max_features': None,
						'bootstrap': None,
						'min_samples_leaf': None,
						'min_samples_split': None,
						'dataset_name_path': "-",
						'running_time': None,
						'CurrentDate': None,
						'stringValues': None
					}
					hyperparameters['stringValues'] = line
					line_split = line.split(",")
					for attributes in line_split:
						attr = attributes.split(":")
						if attr[0] == 'bootstrap':
							# bool conversion
							hyperparameters[attr[0]] = ast.literal_eval(attr[1])
						elif (attr[0] == 'n_estimators') or (attr[0] == 'max_depth') or (attr[0] == 'min_samples_leaf') or (attr[0] == 'min_samples_split'):
							# int conversion
							if(attr[1] != 'None'):
								hyperparameters[attr[0]] = int(attr[1])
							else:
								hyperparameters[attr[0]] = attr[1]
						elif (attr[0] == 'accuracy_score' or attr[0] == 'running_time'):
							# float conversion
							if(attr[1] != 'None'):
								hyperparameters[attr[0]] = float(attr[1])
							else:
								hyperparameters[attr[0]] = attr[1]
						else:
							hyperparameters[attr[0]] =attr[1]

					BestModel_list[i] = hyperparameters
					i = i + 1

				f.close()
				return BestModel_list
		except Exception as e:
			logging.info('Error in read_Random_Forest_BestModel')
			logging.error("Exception occurred in read_Random_Forest_BestModel", exc_info=True)
			raise e
		return None


	# save best parameters in custom logging file - this function is usually called during the hyperparameter optimization
	def save_support_vector_machine_BestModel(self,parammeters):
		try:
			now = datetime.datetime.now()
			currentDate =",CurrentDate:"+now.strftime("%Y-%m-%d %H-%M")+"\n"

			parammeters=parammeters+currentDate
			f = open(self.SVM_filename, "a+")
			f.write(parammeters)
			f.close()
		except Exception as e:
			logging.info('Error in save_support_vector_machine_BestModel')
			logging.error("Exception occurred in save_support_vector_machine_BestModel", exc_info=True)
			raise e
		return None

	# read best parameters from custom logging file
	def read_support_vector_machine_BestModel(self):
		try:
			BestModel_list = {}
			i = 1
			with open(self.SVM_filename) as f:
				for line in f:
					hyperparameters = {
						'accuracy_score': None,
						'kernel': None,
						'C': None,
						'degree': None,
						'gamma': None,
						'coef0': None,
						'tol': None,
						'dataset_name_path': "-",
						'running_time': None,
						'CurrentDate': None
					}

					line_split = line.split(",")
					for attributes in line_split:
						attr = attributes.split(":")
						if (attr[0] == 'C') or (attr[0] == 'degree') or (attr[0] == 'coef0'):
							# int conversion
							if(attr[1] != 'None' and attr[1] != 'auto_deprecated'):
								hyperparameters[attr[0]] = float(attr[1])
							else:
								hyperparameters[attr[0]] = attr[1]
						elif (attr[0] == 'gamma' or attr[0] == 'tol'):
							# float conversion
							if(attr[1] != 'None'  and attr[1] != 'auto_deprecated'):
								hyperparameters[attr[0]] = float(attr[1])
							else:
								hyperparameters[attr[0]] = attr[1]
						else:
							hyperparameters[attr[0]] =attr[1]

					BestModel_list[i] = hyperparameters
					i = i + 1

				f.close()
			# BestModel_list = {v: k for k, v in BestModel_list.items()} to reverse the dictionary
			return BestModel_list
		except Exception as e:
			logging.info('Error in read_support_vector_machine_BestModel')
			logging.error("Exception occurred in read_support_vector_machine_BestModel", exc_info=True)
			raise e
		return None




	# save best parameters in custom logging file - this function is usually called during the hyperparameter optimization
	def save_knn_BestModel(self,parammeters):
		try:
			now = datetime.datetime.now()
			currentDate =",CurrentDate:"+now.strftime("%Y-%m-%d %H-%M")+"\n"

			parammeters=parammeters+currentDate
			f = open(self.KNN_filename, "a+")
			f.write(parammeters)
			f.close()
		except Exception as e:
			logging.info('Error in save_knn_BestModel')
			logging.error("Exception occurred in save_knn_BestModel", exc_info=True)
			raise e
		return None


	# read best parameters from custom logging file
	def read_knn_BestModel(self):
		try:
			BestModel_list = {}
			i = 1
			with open(self.KNN_filename) as f:
				for line in f:
					hyperparameters = {
						'accuracy_score': None,
						'n_neighbors': None,
						'weights': None,
						'algorithm': None,
						'p': None,
						'leaf_size': None,
						'dataset_name_path': "-",
						'running_time': None,
						'CurrentDate': None
					}


					line_split = line.split(",")
					for attributes in line_split:
						attr = attributes.split(":")
						if (attr[0] == 'n_neighbors') or (attr[0] == 'leaf_size') or (attr[0] == 'p'):
							# int conversion
							if(attr[1] != 'None'):
								hyperparameters[attr[0]] = int(attr[1])
							else:
								hyperparameters[attr[0]] = attr[1]
						else:
							hyperparameters[attr[0]] =attr[1]

					BestModel_list[i] = hyperparameters
					i = i + 1

				f.close()
			# BestModel_list = {v: k for k, v in BestModel_list.items()} to reverse the dictionary
			return BestModel_list
		except Exception as e:
			logging.info('Error in read_support_vector_machine_BestModel')
			logging.error("Exception occurred in read_support_vector_machine_BestModel", exc_info=True)
			raise e
		return None




	# save best parameters in custom logging file - this function is usually called during the hyperparameter optimization
	def save_logReg_BestModel(self,parammeters):
		try:
			now = datetime.datetime.now()
			currentDate =",CurrentDate:"+now.strftime("%Y-%m-%d %H-%M")+"\n"

			parammeters=parammeters+currentDate
			f = open(self.LogReg_filename, "a+")
			f.write(parammeters)
			f.close()
		except Exception as e:
			logging.info('Error in save_logReg_BestModel')
			logging.error("Exception occurred in save_logReg_BestModel", exc_info=True)
			raise e
		return None



	# read best parameters from custom logging file
	def read_logReg_BestModel(self):
		try:
			BestModel_list = {}
			i = 1
			with open(self.LogReg_filename) as f:
				for line in f:
					hyperparameters = {
						'accuracy_score': None,
						'penalty': None,
						'dual': None,
						'solver': None,
						'C': None,
						'multi_class': None,
						'max_iter': None,
						'tol': None,
						'dataset_name_path': "-",
						'running_time': None,
						'CurrentDate': None
					}

					line_split = line.split(",")
					for attributes in line_split:
						attr = attributes.split(":")
						if (attr[0] == 'max_iter'):
							# int conversion
							if(attr[1] != 'None'):
								hyperparameters[attr[0]] = int(attr[1])
							else:
								hyperparameters[attr[0]] = attr[1]
						elif (attr[0] == 'tol' or attr[0] == 'C'):
						# float conversion
							if (attr[1] != 'None'):
								hyperparameters[attr[0]] = float(attr[1])
							else:
								hyperparameters[attr[0]] = attr[1]
						elif attr[0] == 'dual':
							# bool conversion
							hyperparameters[attr[0]] = ast.literal_eval(attr[1])
						else:
							hyperparameters[attr[0]] =attr[1]

					BestModel_list[i] = hyperparameters
					i = i + 1

				f.close()
			# BestModel_list = {v: k for k, v in BestModel_list.items()} to reverse the dictionary
			return BestModel_list
		except Exception as e:
			logging.info('Error in read_logReg_BestModel')
			logging.error("Exception occurred in read_logReg_BestModel", exc_info=True)
			raise e
		return None






	# save best parameters in custom logging file - this function is usually called during the hyperparameter optimization
	def save_DecTree_BestModel(self,parammeters):
		try:
			now = datetime.datetime.now()
			currentDate =",CurrentDate:"+now.strftime("%Y-%m-%d %H-%M")+"\n"

			parammeters=parammeters+currentDate
			f = open(self.DecTree_filename, "a+")
			f.write(parammeters)
			f.close()
		except Exception as e:
			logging.info('Error in save_DecTree_BestModel')
			logging.error("Exception occurred in save_DecTree_BestModel", exc_info=True)
			raise e
		return None

	# read best parameters from custom logging file
	def read_DecTree_BestModel(self):
		try:
			BestModel_list = {}
			i = 1
			with open(self.DecTree_filename) as f:
				for line in f:
					hyperparameters = {
						'accuracy_score': None,
						'splitter': None,
						'criterion': None,
						'max_depth': None,
						'max_features': None,
						'min_samples_leaf': None,
						'min_samples_split': None,
						'dataset_name_path': "-",
						'running_time': None,
						'CurrentDate': None,
						'stringValues': None
					}
					hyperparameters['stringValues'] = line
					line_split = line.split(",")
					for attributes in line_split:
						attr = attributes.split(":")
						if (attr[0] == 'max_depth'):
							# int conversion
							if(attr[1] != 'None'):
								hyperparameters[attr[0]] = int(attr[1])
							else:
								hyperparameters[attr[0]] = attr[1]
						elif (attr[0] == 'max_features'):
							if(isinstance(attr[1], str)):
								hyperparameters[attr[0]] = str(attr[1])
							elif(isinstance(attr[1], int)):
								hyperparameters[attr[0]] = int(attr[1])
							elif(isinstance(attr[1], float)):
								hyperparameters[attr[0]] = float(attr[1])
							else:
								hyperparameters[attr[0]] = attr[1]

						elif (attr[0] == 'min_samples_split'):
							# float conversion
							attr[1] = float(attr[1])
							if (attr[1] > 1):
								hyperparameters[attr[0]] = int(attr[1])
							elif (attr[1] <= 1):
								hyperparameters[attr[0]] = float(attr[1])
							else:
								hyperparameters[attr[0]] = attr[1]

						elif (attr[0] == 'min_samples_leaf'):
							# float conversion
							if (attr[1] != 'None'):
								hyperparameters[attr[0]] = float(attr[1])
							else:
								hyperparameters[attr[0]] = attr[1]
						else:
							hyperparameters[attr[0]] =attr[1]

					BestModel_list[i] = hyperparameters
					i = i + 1

				f.close()
			# BestModel_list = {v: k for k, v in BestModel_list.items()} to reverse the dictionary
			return BestModel_list
		except Exception as e:
			logging.info('Error in read_DecTree_BestModel')
			logging.error("Exception occurred in read_DecTree_BestModel", exc_info=True)
			raise e
		return None
