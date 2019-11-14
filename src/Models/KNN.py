import datetime
from sklearn import neighbors
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import time
import logging

path = '../'

# parameterrs string
k_str = 'n_neighbors'
weigh_str = 'weights'
algo_str = 'algorithm'
leaf_str = 'leaf_size'
p_str = 'p'
dataset_str = 'dataset_name_path'
AcuScore_str = 'accuracy_score'
runTime_str = 'running_time'




n_neighbors = [1, 26]
weights = ['uniform', 'distance']
algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']


def setup_logger(name, log_file, level=logging.INFO):
	formatter = logging.Formatter('%(asctime)s [%(filename)s: %(funcName)s - %(lineno)d] - %(message)s', datefmt='%d-%b-%y %H:%M:%S)')
	handler = logging.FileHandler(log_file)
	handler.setFormatter(formatter)
	logger = logging.getLogger(name)
	logger.setLevel(level)
	logger.addHandler(handler)
	return logger


# K nearest neighbor classification
class KNN_class:
	n_neighbors = None
	weights = None
	algorithm = None
	p = None
	model = None

	X_data = None
	y_data = None
	Xtest_data = None
	ytest_data = None
	feature_names = None
	target_names = None

	accuracy_score = None
	y_predicted = None
	optimized_y_predicted = None

	log_BestModel_1 = None
	logger = None
	dataset_name_path = None


	parameter_sets = [
		{
			'n_neighbors': [25],
			'weights': ['distance'],
			'algorithm': ['kd_tree'],
			'n_jobs': [-1],
			'leaf_size': [30],
			'p': [3]
		},
		{
			'n_neighbors': [12, 16, 20, 23, 25],
			'weights': ['uniform', 'distance'],
			'algorithm': ['ball_tree', 'kd_tree', 'brute'],
			'leaf_size': [5, 8, 32, 64, 256, 512],
			'p': [1, 2, 3]
		},
		{
			'n_neighbors': [100, 166, 278, 464, 774],
			'weights': ['uniform', 'distance'],
			'algorithm': ['ball_tree', 'kd_tree', 'brute'],
			'leaf_size': [4, 16, 64, 128, 512],
			'p': [1, 2, 3]
		},
		{
			'n_neighbors': [60, 75, 90, 100, 127],
			'weights': ['uniform', 'distance'],
			'algorithm': ['ball_tree', 'kd_tree', 'brute'],
			'leaf_size': [4, 16, 64, 128, 512],
			'p': [1, 2, 3]
		},
		{
			'n_neighbors': [1291],
			'weights': ['uniform', 'distance'],
			'algorithm': ['ball_tree', 'kd_tree', 'brute'],
			'leaf_size': [4, 32, 64, 256, 512],
			'p': [1, 2, 3]
		},
		{
			'n_neighbors': [5994],
			'weights': ['uniform', 'distance'],
			'algorithm': ['ball_tree', 'kd_tree', 'brute'],
			'leaf_size': [4, 32, 64, 256, 512],
			'p': [1, 2, 3]
		},
		{
			'n_neighbors': [10000],
			'weights': ['uniform', 'distance'],
			'algorithm': ['ball_tree', 'kd_tree', 'brute'],
			'leaf_size': [32, 64, 256],
			'p': [1, 2, 3]
		},
		{
			'n_neighbors': [10000],
			'weights': ['uniform', 'distance'],
			'algorithm': ['ball_tree', 'kd_tree', 'brute'],
			'leaf_size': [4, 512],
			'p': [1, 2, 3]
		},
		{
			'n_neighbors': [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
			'weights': ['uniform', 'distance'],
			'algorithm': ['ball_tree', 'kd_tree', 'brute'],
			'n_jobs': [-1],
			'leaf_size': [1, 2, 4, 8, 16, 32, 128, 256],
			'p': [1, 2, 3]
		},
		{
			'n_neighbors': [3593],
			'weights': ['uniform', 'distance'],
			'algorithm': ['ball_tree', 'kd_tree', 'brute'],
			'n_jobs': [-1],
			'leaf_size': [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
			'p': [1, 2, 3]
		},
		{
			'n_neighbors': [2154],
			'weights': ['uniform', 'distance'],
			'algorithm': ['ball_tree', 'kd_tree', 'brute'],
			'n_jobs': [-1],
			'leaf_size': [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
			'p': [1, 2, 3]
		}
		# {
		# 	'n_neighbors': [774, 1291, 2154, 3593, 5994, 10000],
		# 	'weights': ['uniform', 'distance'],
		# 	'algorithm': ['ball_tree', 'kd_tree', 'brute'],
		# 	'leaf_size': [4, 32, 64, 256, 512],
		# 	'p': [1, 2, 3]
		# },
		# {
		# 	'n_neighbors': [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
		# 	'weights': ['uniform', 'distance'],
		# 	'algorithm': ['ball_tree', 'kd_tree', 'brute'],
		# 	'n_jobs': [-1],
		# 	'leaf_size': [1, 2, 4, 8, 16, 32, 128, 256],
		# 	'p': [1, 2, 3]
		# },
		# {
		# 	'n_neighbors': [100, 166, 278, 464, 774, 1291, 2154, 3593, 5994, 10000],
		# 	'weights': ['uniform', 'distance'],
		# 	'algorithm': ['ball_tree', 'kd_tree', 'brute'],
		# 	'n_jobs': [-1],
		# 	'leaf_size': [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
		# 	'p': [1, 2, 3]
		# }

	]


	def __init__(self, Log_BestModel, inX_data, iny_data, inXtest_data, inytest_data, infeature_name, intarget_name, dataset_name_path, in_weights='uniform', in_neighbors=10, in_algorithm='auto'):
		try:
			logger = setup_logger('KNNTraining', path + 'Logging/KNNTraining.log')
			self.logger = logger
			self.dataset_name_path = dataset_name_path


			self.logger.info("\n\n\n\n")
			self.logger.info("\n\n\n\n")
			self.logger.info('=============== Initialize the KNN_class variable ==================')
			self.logger.info("===== Working on following data set: ======")
			self.logger.info(dataset_name_path)
			currentDate = " CurrentDate: " + datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
			self.logger.info(currentDate)
			self.logger.info("\n\n\n\n")

			scaling = MinMaxScaler(feature_range=(-1, 1)).fit(inX_data)
			inX_data = scaling.transform(inX_data)
			inXtest_data = scaling.transform(inXtest_data)

			# set parameter
			self.n_neighbors = in_neighbors
			self.weights = in_weights
			self.algorithm = in_algorithm
			self.p = 1

			# set data
			self.X_data = inX_data
			self.y_data = iny_data
			self.Xtest_data = inXtest_data
			self.ytest_data = inytest_data
			self.feature_names = infeature_name
			self.target_names = intarget_name

			# set model
			self.model = neighbors.KNeighborsClassifier(n_neighbors=in_neighbors, weights=in_weights, algorithm=in_algorithm)
			self.log_BestModel_1 = Log_BestModel
			print("------------------------------ KNN_class Initialized -------------------------")

		except Exception as e:
			self.logger.error("Exception occurred in __init__", exc_info=True)
			raise e

	def train_model(self):
		try:
			self.model.fit(self.X_data, self.y_data)

		except Exception as e:
			raise e
		print("------------------------------ Model trained  ------------------------")

	def predict(self):
		try:
			self.y_predicted = self.model.predict(self.Xtest_data)
		except Exception as e:
			raise e
		print("------------------------------ Model predict  ------------------------")

	def calculate_accuracy_score(self):
		try:
			print("--------------------------- Accuracy score of model -------------------------------------------")
			self.accuracy_score = metrics.accuracy_score(self.ytest_data, self.y_predicted)
			print(self.accuracy_score)
			self.logger.info('Single Accuracy score of model: '+str(self.accuracy_score))
		except Exception as e:
			raise e


	def classification_report(self):
		try:
			print("--------------------------- classification report --------------------------------------------")
			print(metrics.classification_report(self.ytest_data, self.y_predicted, target_names=self.target_names))
		except Exception as e:
			raise e


	def confusion_matrix(self):
		try:
			print("--------------------------- confusion matrix -------------------------------------------")
			print(metrics.confusion_matrix(self.ytest_data, self.y_predicted))
		except Exception as e:
			raise e



	def hyperparameter_optimization_search(self,gridSearch=False, in_iter = 20, in_random_state = 77, in_cv=5, in_num_jobs=-1, in_verbosity=200):
		try:
			i = 0
			if (gridSearch):
				self.logger.info('------------------------- Grid-Search ------------------------')
			else:
				self.logger.info('------------------------- Random-Search ------------------------')

			for hyperparameters in self.parameter_sets:

				i += 1
				currentDate = "    CurrentDate: " + datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
				searchIteration = "    Search iteration: "+ str(i)
				self.logger.info("\n\n")

				self.logger.info('------------------------- Hyperparameter_optimization_new ------------------------')
				self.logger.info("--------------------------------------------------")
				self.logger.info(searchIteration)
				self.logger.info(currentDate)
				self.logger.info("--------------------------------------------------")
				self.logger.info("\n\n")

				print('------------------------- Hyperparameter optimization -new- ------------------------')
				self.logger.info(hyperparameters)
				print(hyperparameters)
				start = time.time()
				optimized_model = None
				if (gridSearch):
					optimized_model = GridSearchCV(self.model, hyperparameters, cv=in_cv, n_jobs=in_num_jobs,
											   verbose=in_verbosity)
				else:
					optimized_model = RandomizedSearchCV(estimator=self.model, param_distributions=hyperparameters, n_iter=in_iter,
											 cv=in_cv,
											 verbose=in_verbosity, random_state=in_random_state, n_jobs=in_num_jobs)

				optimized_model.fit(self.X_data, self.y_data)

				self.optimized_y_predicted = optimized_model.predict(self.Xtest_data)

				end = time.time()
				running_time = end - start
				print("Total optimization time in s:", str(running_time))

				self.process_trained_model_information(optimized_model,running_time)

				print("*********----------------- hyperparameter optimization end  ---------------*********")
				self.logger.info("---------------- hyperparameter-optimization-end  ----------------")

		except Exception as e:
			self.logger.error("Exception occurred in hyperparameter_optimization", exc_info=True)
			raise e




	def process_trained_model_information(self, optimized_model, running_time):
		try:
				self.logger.info('-----------------All_iterated_models----------------')
				print()
				print('-----------------All iterated models----------------')
				print()

				cvres = optimized_model.cv_results_
				for mean_score, std_score, params in zip(cvres["mean_test_score"], cvres["std_test_score"],
														 cvres["params"]):
					print(mean_score, std_score, params)
					stringValue = " mean_score:" + str(mean_score), " std_score:" + str(std_score), " params:" + str(
						params)
					self.logger.info(stringValue)


				# Best Model info
				self.logger.info('----------------Best_Model----------------')
				print()
				print('-----------------Best Model ----------------')
				print()

				optimized_accuracy_score = metrics.accuracy_score(self.ytest_data, self.optimized_y_predicted)

				k_value = optimized_model.best_estimator_.get_params()[k_str]
				w_value = optimized_model.best_estimator_.get_params()[weigh_str]
				algo_value = optimized_model.best_estimator_.get_params()[algo_str]
				leaf_value = optimized_model.best_estimator_.get_params()[leaf_str]
				p_value = optimized_model.best_estimator_.get_params()[p_str]
				ac_score = optimized_accuracy_score * 100

				print("(\n " + AcuScore_str + ": ", ac_score,
					  ",\n " + k_str + ": ", k_value,
					  ",\n " + weigh_str + ": ", w_value,
					  ",\n " + algo_str + ": ", algo_value,
					  ",\n " + p_str + ": ", p_value,
					  ",\n " + leaf_str + ": ", leaf_value,
					  ",\n " + dataset_str + ": ", self.dataset_name_path,
					  ",\n " + runTime_str + ": ", running_time, "\n)")

				parameters = AcuScore_str + ":" + str(ac_score) \
							 + "," + k_str + ":" + str(k_value) \
							 + "," + weigh_str + ":" + str(w_value)\
							 + "," + algo_str + ":" + str(algo_value)\
							 + "," + p_str + ":" + str(p_value)\
							 +"," + leaf_str + ":" + str(leaf_value)\
							 +"," + dataset_str + ":" + str(self.dataset_name_path)\
							 + "," + runTime_str + ":" + str(running_time)


				self.logger.info(parameters)
				self.log_BestModel_1.save_knn_BestModel(parameters)

				# precision_recall_curve
				print("*********----------------- precision_recall_curve ---------------*********")
				precision, recall, thresholds = metrics.precision_recall_curve(self.ytest_data,
																			   self.optimized_y_predicted)

				print("Precision:" + str(precision)
					  + "    Recall:" + str(recall)
					  + "     Thresholds:" + str(thresholds))

				precision_recall = "Precision:" + str(precision) \
								   + ",   Recall:" + str(recall) \
								   + ",   Thresholds:" + str(thresholds)

				self.logger.info('---------------- Precision_Recall ----------------')
				self.logger.info(precision_recall)

				# confusion report
				# confusion matrix
				self.calculate_score_and_report(optimized_model)

				# print best parameter after tuning
				print("print best parameter after tuning:")
				print(optimized_model.best_params_)
				self.logger.info("----------------  print best parameter after tuning: ----------------")
				self.logger.info(optimized_model.best_params_)

				# print how our model looks after hyper-parameter tuning
				print("print how our best model looks after hyper-parameter tuning:")
				print(optimized_model.best_estimator_)
				self.logger.info("---------------- print how our best model looks after hyper-parameter tuning: ----------------")
				self.logger.info(optimized_model.best_estimator_)


				# print
				print("print optimized_model.cv_results_.keys:")
				print(sorted(optimized_model.cv_results_.keys()))
				self.logger.info("---------------- optimized_model.cv_results_.keys: ----------------")
				self.logger.info(sorted(optimized_model.cv_results_.keys()))



		except Exception as e:
			self.logger.error("Exception occurred in hyperparameter_optimization", exc_info=True)
			raise e


		return None

	def calculate_score_and_report(self, Best_Model):

		self.logger.info("---------------- Grid_scores_on_development_set ----------------")

		print("------------------------------------------------------------------")
		print("Grid scores on development set:")
		print()
		means = Best_Model.cv_results_['mean_test_score']
		stds = Best_Model.cv_results_['std_test_score']
		self.logger.info(means)
		self.logger.info(stds)

		for mean, std, params in zip(means, stds, Best_Model.cv_results_['params']):
			print("%0.3f (+/-%0.03f) for %r"
				  % (mean, std * 2, params))
			str_val="Mean:"+str(mean)+" std:"+str(std*2)+" params:"+str(params)
			self.logger.info(str_val)
		print()

		print("Detailed classification report:")
		print()
		print("The model is trained on the full development set.")
		print("The scores are computed on the full evaluation set.")
		print()

		self.logger.info(" ")
		self.logger.info("Detailed classification report:")
		self.logger.info("The model is trained on the full development set.")
		self.logger.info("The scores are computed on the full evaluation set.")
		self.logger.info(" ")

		# y_true, y_pred = self.ytest_data, Best_Model.predict(self.Xtest_data)
		print(metrics.classification_report(self.ytest_data, self.optimized_y_predicted))
		self.logger.info(metrics.classification_report(self.ytest_data, self.optimized_y_predicted))

		print("---------------- Confusion matrix ----------------")
		self.logger.info("---------------- Confusion_matrix ----------------")

		print(metrics.confusion_matrix(self.ytest_data, self.optimized_y_predicted))
		self.logger.info(metrics.confusion_matrix(self.ytest_data, self.optimized_y_predicted))


		print()
		print("------------------------------------------------------------------")
		return None

	def display_all_bestModel(self):
		print("--------------------------- display all bestModel -------------------------------------------")
		try:
			bestModel_List = self.log_BestModel_1.read_knn_BestModel()
			for index in bestModel_List:
				print(index)
				print("(\n " + AcuScore_str + ": ", bestModel_List[index][AcuScore_str],
					  ",\n " + k_str + ": ", bestModel_List[index][k_str],
					  ",\n " + weigh_str + ": ", bestModel_List[index][weigh_str],
					  ",\n " + algo_str + ": ", bestModel_List[index][algo_str],
					  ",\n " + leaf_str + ": ", bestModel_List[index][leaf_str],
					  ",\n " + p_str + ": ", bestModel_List[index][p_str],
					  ",\n " + dataset_str + ": ", bestModel_List[index][dataset_str],
					  ",\n " + runTime_str + ": ", bestModel_List[index][runTime_str],
					  ",\n CurrentDate: ", bestModel_List[index]['CurrentDate'],
					  "\n)")

		except Exception as e:
			self.logger.error("Exception occurred in display_all_bestModel", exc_info=True)
			raise e

		self.logger.info('display all bestModel')

		return None
