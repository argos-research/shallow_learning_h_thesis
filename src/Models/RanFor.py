# coding=utf-8
import datetime
from sklearn import ensemble
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import time
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.metrics import confusion_matrix



# n_estimators
# criterion
# max_depth
max_features = ["auto", "sqrt", "log2"]
bootstrap = [False, True]
path = '../'


def setup_logger(name, log_file, level=logging.INFO):
	formatter = logging.Formatter('%(asctime)s [%(filename)s: %(funcName)s - %(lineno)d] - %(message)s', datefmt='%d-%b-%y %H:%M:%S)')
	handler = logging.FileHandler(log_file)
	handler.setFormatter(formatter)
	logger = logging.getLogger(name)
	logger.setLevel(level)
	logger.addHandler(handler)
	return logger


# Random Forest
class Ranfor_class:
	n_estimators = None
	criterion = None
	max_depth = None
	max_features = None
	bootstrap = None
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


	parameter_sets= [
		{
			'n_estimators': [100],
			'max_depth': [25],
			'min_samples_split': [8],
			'min_samples_leaf': [5]
		},
		{
			'n_estimators': [100, 300, 500, 800, 1200],
			'max_depth': [5, 8, 15, 25, 30],
			'min_samples_split': [2, 5, 8, 50],
			'min_samples_leaf': [1, 5, 9]
		},
		{   'bootstrap': [False],
			'n_estimators': [4, 16, 64, 800, 1200],
			'max_depth': [ 70, 85, 100, None],
			'min_samples_leaf': [3, 6, 9],
			'min_samples_split': [10, 15, 100]
		},

		{	'criterion': ['entropy'],
			'n_estimators': [100, 500, 1200],
			'max_depth': [5, 15, 25, 30],
			'min_samples_split': [2, 8, 50],
			'min_samples_leaf': [1, 5, 9],
		},
		{	 'criterion': ['entropy'],
			 'bootstrap': [False],
			 'n_estimators': [4, 64, 750, 1000],
			 'max_depth': [70, 90, 100],
			 'min_samples_split': [10, 15, 100],
			 'min_samples_leaf': [3, 6, 9],
		 }
	]

	def __init__(self, Log_BestModel, inX_data, iny_data, inXtest_data, inytest_data, infeature_name, intarget_name, dataset_name_path, in_estimators=100, in_criterion='gini', in_max_depth=None, in_max_features='auto', in_bootstrap=True):
		try:
			logger = setup_logger('RanForTraining', path + 'Logging/RanForTraining.log')
			self.logger = logger
			self.dataset_name_path = dataset_name_path

			self.logger.info("\n\n\n\n")
			self.logger.info("\n\n\n\n")
			self.logger.info('=============== Initialize the Ranfor_class variable ==================')
			self.logger.info("===== Working on following data set: ======")
			self.logger.info(dataset_name_path)
			currentDate = " CurrentDate: " + datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
			self.logger.info(currentDate)
			self.logger.info("\n\n\n\n")

			# set parameter
			self.n_estimators = in_estimators
			self.criterion = in_criterion
			self.max_depth = in_max_depth
			self.max_features = in_max_features
			self.bootstrap = in_bootstrap

			# set data
			self.X_data = inX_data
			self.y_data = iny_data
			self.Xtest_data = inXtest_data
			self.ytest_data = inytest_data
			self.feature_names = infeature_name
			self.target_names = intarget_name

			# set model
			self.model = ensemble.RandomForestClassifier(n_estimators=in_estimators, criterion=in_criterion, max_depth=in_max_depth, max_features=in_max_features, bootstrap=in_bootstrap)
			self.log_BestModel_1 = Log_BestModel

			self.logger.info('Initialization Done')
			print("------------------------------ Ranfor_class Initialized ----------------------")
		except Exception as e:
			self.logger.error("Exception occurred in __init__", exc_info=True)
			raise e


	def train_model(self):
		try:
			self.model.fit(self.X_data, self.y_data)
		except Exception as e:
			raise e
		print("------------------------------ Model trained  ------------------------")



	def confusion_matrix_graph(self):
		try:

			self.model.fit(self.X_data, self.y_data)
			self.predict()

			y_true = self.ytest_data
			y_pred = self.y_predicted
			filename = "confusion_matrix.png"
			labels = self.target_names
			ymap = None
			figsize = (2, 2)
			if ymap is not None:
				y_pred = [ymap[yi] for yi in y_pred]
				y_true = [ymap[yi] for yi in y_true]
				labels = [ymap[yi] for yi in labels]

			cm = metrics.confusion_matrix(self.ytest_data, self.y_predicted)
			# cm = metrics.confusion_matrix(self.ytest_data, self.y_predicted,labels=labels)
			# cm = confusion_matrix(y_true, y_pred, labels=labels)
			cm_sum = np.sum(cm, axis=1, keepdims=True)
			cm_perc = cm / cm_sum.astype(float) * 100
			annot = np.empty_like(cm).astype(str)
			nrows, ncols = cm.shape
			for i in range(nrows):
				for j in range(ncols):
					c = cm[i, j]
					p = cm_perc[i, j]
					if i == j:
						s = cm_sum[i]
						annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
					elif c == 0:
						annot[i, j] = ''
					else:
						annot[i, j] = '%.1f%%\n%d' % (p, c)
			cm = pd.DataFrame(cm, index=labels, columns=labels)
			cm.index.name = 'Actual'
			cm.columns.name = 'Predicted'
			# fig, ax = plt.subplots(figsize=figsize)
			# sns.heatmap(cm, annot=annot, fmt='', ax=ax)
			# plt.savefig(filename)
		except Exception as e:
			raise e
		print("------------------------------ Model trained  ------------------------")




	def training_model_n_feature_importance(self):
		try:
			f = open("../Evaluation/RandForFeatureImportance.txt", "a+")

			now = datetime.datetime.now()
			currentDate = ",CurrentDate:" + now.strftime("%Y-%m-%d %H-%M") + "\n"
			f.write("\n\n------------------------------ PV6 New training_model_n_feature_importance Execution Started --------------------------------")
			f.write("\n---------- "+self.dataset_name_path+" -------------")
			f.write("\n---------- "+str(currentDate)+" -------------")
			f.write("\n---------- " + str(self.dataset_name_path) + " -------------")

			# p6
			# # {'n_estimators': 300, 'min_samples_split': 8, 'min_samples_leaf': 1, 'max_depth': 25}
			# self.model = ensemble.RandomForestClassifier(n_estimators=300,
			# 											 max_depth=25,
			# 											 min_samples_split=8,
			# 											 min_samples_leaf=1)

			# R3
			# Acc
			self.model = ensemble.RandomForestClassifier(n_estimators=500,
														 max_depth=70,
													 min_samples_split=15,
														 min_samples_leaf=3,
														 bootstrap=False,
														 criterion='entropy',
														 max_features='auto')

			self.model.fit(self.X_data, self.y_data)
			self.model.score(self.Xtest_data, self.ytest_data)

			feature_importances = pd.DataFrame(self.model.feature_importances_, index=self.X_data.columns, columns=['importance']).sort_values('importance', ascending=False)
			print(feature_importances)
			f.write(feature_importances.to_string())

			self.predict()
			optimized_accuracy_score = metrics.accuracy_score(self.ytest_data, self.y_predicted)
			ac_score = optimized_accuracy_score * 100
			print("\nac_score")
			print(ac_score)

			f.write("\nac_score")
			f.write(str(ac_score))

			f.write("\nfeature_names")
			f.write(str(self.feature_names))

		except Exception as e:
			self.logger.error("Exception occurred in train_model", exc_info=True)
			raise e
		f.write("\n\n Execution ended")
		f.close()
		self.logger.info('Single Model trained')
		print("------------------------------ Model trained  --------------------------------")


	def predict(self):
		try:
			self.y_predicted = self.model.predict(self.Xtest_data)
		except Exception as e:
			self.logger.error("Exception occurred in predict", exc_info=True)
			raise e
		print("------------------------------ Model predict  --------------------------------")
		self.logger.info('Single Model prediction')

	def calculate_accuracy_score(self):
		try:
			print("------------------------------ Accuracy score of model -----------------------")
			self.accuracy_score = metrics.accuracy_score(self.ytest_data, self.y_predicted)
			print(self.accuracy_score)
			self.logger.info('Single Accuracy score of model: '+str(self.accuracy_score))
		except Exception as e:
			self.logger.error("Exception occurred in calculate_accuracy_score", exc_info=True)
			raise e

	def classification_report(self):
		try:
			print("----------------------------- classification report --------------------------")
			print(metrics.classification_report(self.ytest_data, self.y_predicted, target_names=self.target_names))

			self.logger.info('classification report')
		except Exception as e:
			self.logger.error("Exception occurred in classification_report", exc_info=True)
			raise e

	def confusion_matrix(self):
		try:
			print("------------------------------ confusion matrix ------------------------------")
			print(metrics.confusion_matrix(self.ytest_data, self.y_predicted))
			self.logger.info('confusion matrix')
		except Exception as e:
			self.logger.error("Exception occurred in confusion_matrix", exc_info=True)
			raise e

	def display_all_bestModel(self):
		print("------------------------------ display all bestModel -------------------------")
		try:
			bestModel_List = self.log_BestModel_1.read_Random_Forest_BestModel()
			for index in bestModel_List:
				print(index)
				print("(\n accuracy_score: ", bestModel_List[index]['accuracy_score'],
					  ",\n n_estimators: ",bestModel_List[index]['n_estimators'],
					  ",\n criterion: ", bestModel_List[index]['criterion'],
					  ",\n max_depth: ", bestModel_List[index]['max_depth'],
					  ",\n max_features: ", bestModel_List[index]['max_features'],
					  ",\n bootstrap: ", bestModel_List[index]['bootstrap'],
					  ",\n min_samples_leaf: ", bestModel_List[index]['min_samples_leaf'],
					  ",\n min_samples_split: ",bestModel_List[index]['min_samples_split'],
					  ",\n dataset_name_path: ",bestModel_List[index]['dataset_name_path'],
					  ",\n running_time: ",bestModel_List[index]['running_time'],
					  ",\n CurrentDate: ",bestModel_List[index]['CurrentDate'],
					  "\n)")

		except Exception as e:
			self.logger.error("Exception occurred in display_all_bestModel", exc_info=True)
			raise e

		self.logger.info('display all bestModel')

		return None

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

				est = optimized_model.best_estimator_.get_params()["n_estimators"]
				cri = optimized_model.best_estimator_.get_params()['criterion']
				dept = optimized_model.best_estimator_.get_params()['max_depth']
				max_fe = optimized_model.best_estimator_.get_params()['max_features']
				bstp = optimized_model.best_estimator_.get_params()['bootstrap']
				min_l = optimized_model.best_estimator_.get_params()['min_samples_leaf']
				min_sp = optimized_model.best_estimator_.get_params()['min_samples_split']
				ac_score = optimized_accuracy_score * 100

				print("(\n accuracy_score: ", ac_score
					  , ",\n n_estimators: ", est
					  ,",\n criterion: ", cri
					  , ",\n max_depth: ", dept
					  , ",\n max_features: ",max_fe
					  , ",\n bootstrap: ", bstp
					  , ",\n min_samples_leaf: ", min_l
					  , ",\n min_samples_split: ", min_sp
					  , ",\n dataset_name_path: ", self.dataset_name_path
					  ,",\n running_time: ", running_time, "\n)")

				parameters = "accuracy_score:"+str(ac_score)\
							 +",n_estimators:"+str(est)\
							 +",criterion:"+str(cri)\
							 +",max_depth:"+str(dept)\
							 +",max_features:"+str(max_fe)\
							 +",bootstrap:"+str(bstp)\
							 +",min_samples_leaf:"+str(min_l)\
							 +",min_samples_split:"+str(min_sp)\
							 +",dataset_name_path:"+str(self.dataset_name_path)\
							 +",running_time:"+str(running_time)

				self.logger.info(parameters)
				self.log_BestModel_1.save_Random_Forest_BestModel(parameters)

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