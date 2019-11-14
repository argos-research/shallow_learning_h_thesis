import datetime
from sklearn import linear_model
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import time
import logging


path = '../'
criterion = ['gini','entropy']
splitter = ['best','random']
max_depth = None
max_features = None
in_min_samples_leaf = None
in_min_samples_split = None


# parameterrs string
split_str = 'splitter'
crit_str = 'criterion'
maxDept_str = 'max_depth'
maxFea_str = 'max_features'
minLeaf_str = 'min_samples_leaf'
minSplit_str = 'min_samples_split'
dataset_str = 'dataset_name_path'
AcuScore_str = 'accuracy_score'
runTime_str = 'running_time'

def setup_logger(name, log_file, level=logging.INFO):
	formatter = logging.Formatter('%(asctime)s [%(filename)s: %(funcName)s - %(lineno)d] - %(message)s', datefmt='%d-%b-%y %H:%M:%S)')
	handler = logging.FileHandler(log_file)
	handler.setFormatter(formatter)
	logger = logging.getLogger(name)
	logger.setLevel(level)
	logger.addHandler(handler)
	return logger

# Decision Tree
class DecTree_class:
	splitter = None
	criterion = None
	max_depth = None
	max_features = None
	min_samples_leaf = None
	min_samples_split = None
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
			'splitter': ['best'],
			'criterion': ['gini'],
			'max_depth': [16],
			'max_features': [10],
			'min_samples_leaf': [3],
			'min_samples_split': [50]
		},
		{
			'splitter': ['best'],
		},
		{
			'criterion': ['gini'],
		},
		{
			'splitter': ['best', 'random'],
			'criterion': ['gini', 'entropy'],
			'max_depth': [None, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
			'max_features': ['auto', 'sqrt', 'log2'],
			'min_samples_leaf': [3, 6, 9],
			'min_samples_split': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2, 4, 8, 16]
		},
		{
			'splitter': ['best', 'random'],
			'criterion': ['gini', 'entropy'],
			'max_depth': [None, 1, 4, 8, 9, 12, 16, 20, 25, 26, 30, 35, 45, 50],
			'max_features': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
			'min_samples_leaf': [3, 6, 9],
			'min_samples_split': [5, 10, 20, 30, 40, 50, 100, 150, 200, 250, 300, 350, 400, 450]
		},
		{
			'splitter': ['best', 'random'],
			'criterion': ['gini', 'entropy'],
			'max_depth': [None, 1, 4, 8, 9, 12, 16, 20, 25, 26, 30, 35, 45, 50],
			'max_features': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
			'min_samples_leaf': [0.1, 0.2, 0.3, 0.4, 0.5],
			'min_samples_split': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2, 4, 8, 16]
		},
		{
			'splitter': ['best', 'random'],
			'criterion': ['gini', 'entropy'],
			'max_depth': [None, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
			'max_features': ['auto', 'sqrt', 'log2'],
			'min_samples_leaf': [0.1, 0.2, 0.3, 0.4, 0.5],
			'min_samples_split': [5, 10, 20, 30, 40, 50, 100, 150, 200, 250, 300, 350, 400, 450]
		}
]

	def __init__(self, Log_BestModel, inX_data, iny_data, inXtest_data, inytest_data, infeature_name, intarget_name, dataset_name_path, in_criterion='gini', in_splitter='best', in_max_depth=None, in_max_features=None, in_min_samples_leaf=1, in_min_samples_split=2):
		try:
			logger = setup_logger('DecTreeTraining', path + 'Logging/DecTreeTraining.log')
			self.logger = logger
			self.dataset_name_path = dataset_name_path

			self.logger.info("\n\n\n\n")
			self.logger.info("\n\n\n\n")
			self.logger.info('=============== Initialize the DecTree_class variable ==================')
			self.logger.info("===== Working on following data set: ======")
			self.logger.info(dataset_name_path)
			currentDate = " CurrentDate: " + datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
			self.logger.info(currentDate)
			self.logger.info("\n\n\n\n")

			# set parameter
			self.criterion = in_criterion
			self.splitter = in_splitter
			self.max_depth = in_max_depth
			self.max_features = in_max_features
			self.min_samples_leaf = in_min_samples_leaf
			self.min_samples_split = in_min_samples_split

			# set data
			self.X_data = inX_data
			self.y_data = iny_data
			self.Xtest_data = inXtest_data
			self.ytest_data = inytest_data
			self.feature_names = infeature_name
			self.target_names = intarget_name

			# set model
			self.model = tree.DecisionTreeClassifier(criterion=in_criterion, splitter=in_splitter, max_depth=in_max_depth, max_features=in_max_features, min_samples_leaf=in_min_samples_leaf, min_samples_split=in_min_samples_split)
			self.log_BestModel_1 = Log_BestModel
			print("------------------------------ DecTree_class Initialized ---------------------")
		except Exception as e:
			self.logger.error("Exception occurred in __init__", exc_info=True)
			raise e

	def train_model(self):
		try:
			# a = svm.SVC(kernel='rbf', gamma='scale')
			# a.fit(self.X_data, self.y_data)
			self.model.fit(self.X_data, self.y_data)
			feature_importances = pd.DataFrame(self.model.feature_importances_, index=self.X_data.columns,
											   columns=['importance']).sort_values('importance', ascending=False)
			print(feature_importances)
		except Exception as e:
			raise e
		print("------------------------------ Model trained  ------------------------")

	def training_model_n_feature_importance(self):
		try:
			f = open("../Evaluation/DecisionTreeImportance.txt", "a+")

			now = datetime.datetime.now()
			currentDate = ",CurrentDate:" + now.strftime("%Y-%m-%d %H-%M") + "\n"
			f.write("\n\n------------------------------ 11111 New training_model_n_feature_importance Execution Started --------------------------------")
			f.write("\n---------- " + self.dataset_name_path + " -------------")
			f.write("\n---------- " + str(currentDate) + " -------------")
			f.write("\n---------- " + str(self.dataset_name_path) + " -------------")


			# accuracy_score: 98.0881813698838, splitter: best, criterion: entropy, max_depth: 40, max_features: auto, min_samples_leaf: 6, min_samples_split: 2, dataset_name_path: Datasets / FinalDatasetR3 / r3__feature_data.csv, running_time: 9.697100400924683
			# # R3 99.3032988986812
			# in_criterion='entropy'
			# in_max_depth =40
			# in_min_samples_leaf=6
			# in_min_samples_split =2
			# in_max_features ='auto'
			# in_splitter ='best'

			# self.model = tree.DecisionTreeClassifier(criterion=in_criterion, splitter=in_splitter,
			# 										 max_depth=in_max_depth, max_features=in_max_features,
			# 										 min_samples_leaf=in_min_samples_leaf,
			# 										 min_samples_split=in_min_samples_split)

			# accuracy_score: 99.3032988986812, splitter: best, criterion: gini, max_depth: None, max_features: None, min_samples_leaf: 1, min_samples_split: 2, dataset_name_path: Datasets / FinalDatasetR3 / r3__feature_data.csv, running_time: 10.557979345321655
			# # R3 99.3032988986812
			in_criterion='gini'
			in_max_depth =None
			in_min_samples_leaf=1
			in_min_samples_split =2
			in_max_features =None
			in_splitter ='best'
			self.model = tree.DecisionTreeClassifier(criterion=in_criterion, splitter=in_splitter,
													 max_depth=in_max_depth, max_features=in_max_features,
													 min_samples_leaf=in_min_samples_leaf,
													 min_samples_split=in_min_samples_split)

			self.model .fit(self.X_data, self.y_data)
			self.model .score(self.Xtest_data, self.ytest_data)

			feature_importances = pd.DataFrame(self.model.feature_importances_, index=self.X_data.columns,
											   columns=['importance']).sort_values('importance', ascending=False)
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


	def hyperparameter_optimization_search(self, gridSearch=False, in_iter=20, in_random_state=77, in_cv=5,
										   in_num_jobs=-1, in_verbosity=200):
		try:

			i = 0
			if (gridSearch):
				self.logger.info('------------------------- Grid-Search ------------------------')
			else:
				self.logger.info('------------------------- Random-Search ------------------------')

			for hyperparameters in self.parameter_sets:

				i += 1
				currentDate = "    CurrentDate: " + datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
				searchIteration = "    Search iteration: " + str(i)
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
					optimized_model = RandomizedSearchCV(estimator=self.model, param_distributions=hyperparameters,
														 n_iter=in_iter,
														 cv=in_cv,
														 verbose=in_verbosity, random_state=in_random_state,
														 n_jobs=in_num_jobs)

				optimized_model.fit(self.X_data, self.y_data)

				self.optimized_y_predicted = optimized_model.predict(self.Xtest_data)

				end = time.time()
				running_time = end - start
				print("Total optimization time in s:", str(running_time))

				self.process_trained_model_information(optimized_model, running_time)

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

			split_value = optimized_model.best_estimator_.get_params()[split_str]
			crit_value = optimized_model.best_estimator_.get_params()[crit_str]
			maxDept_value = optimized_model.best_estimator_.get_params()[maxDept_str]
			maxFea_value = optimized_model.best_estimator_.get_params()[maxFea_str]
			minLeaf_value = optimized_model.best_estimator_.get_params()[minLeaf_str]
			minSplit_value = optimized_model.best_estimator_.get_params()[minSplit_str]
			ac_score = optimized_accuracy_score * 100

			print("(\n " + AcuScore_str + ": ", ac_score,
				  ",\n " + split_str + ": ", split_value,
				  ",\n " + crit_str + ": ", crit_value,
				  ",\n " + maxDept_str + ": ", maxDept_value,
				  ",\n " + maxFea_str + ": ", maxFea_value,
				  ",\n " + minLeaf_str + ": ", minLeaf_value,
				  ",\n " + minSplit_str + ": ", minSplit_value,
				  ",\n " + dataset_str + ": ", self.dataset_name_path,
				  ",\n " + runTime_str + ": ", running_time, "\n)")

			parameters = AcuScore_str + ":" + str(ac_score) \
						 + "," + split_str + ":" + str(split_value) \
						 + "," + crit_str + ":" + str(crit_value) \
						 + "," + maxDept_str + ":" + str(maxDept_value) \
						 + "," + maxFea_str + ":" + str(maxFea_value) \
						 + "," + minLeaf_str + ":" + str(minLeaf_value) \
						 + "," + minSplit_str + ":" + str(minSplit_value) \
						 + "," + dataset_str + ":" + str(self.dataset_name_path) \
						 + "," + runTime_str + ":" + str(running_time)


			self.logger.info(parameters)
			self.log_BestModel_1.save_DecTree_BestModel(parameters)

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
			self.logger.info(
				"---------------- print how our best model looks after hyper-parameter tuning: ----------------")
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
			str_val = "Mean:" + str(mean) + " std:" + str(std * 2) + " params:" + str(params)
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
		print("--------------------------- read_DecTree_BestModel display all bestModel -------------------------------------------")
		self.logger.info("--------------------------- read_DecTree_BestModel display all bestModel -------------------------------------------")
		try:
			bestModel_List = self.log_BestModel_1.read_DecTree_BestModel()
			for index in bestModel_List:
				print(index)
				print("(\n " + AcuScore_str + ": ", bestModel_List[index][AcuScore_str],
					  ",\n " + split_str + ": ", bestModel_List[index][split_str],
					  ",\n " + crit_str + ": ", bestModel_List[index][crit_str],
					  ",\n " + maxDept_str + ": ", bestModel_List[index][maxDept_str],
					  ",\n " + maxFea_str + ": ", bestModel_List[index][maxFea_str],
					  ",\n " + minLeaf_str + ": ", bestModel_List[index][minLeaf_str],
					  ",\n " + minSplit_str + ": ", bestModel_List[index][minSplit_str],
					  ",\n " + dataset_str + ": ", bestModel_List[index][dataset_str],
					  ",\n " + runTime_str + ": ", bestModel_List[index][runTime_str],
					  ",\n CurrentDate: ", bestModel_List[index]['CurrentDate'],
					  "\n)")

		except Exception as e:
			self.logger.error("read_DecTree_BestModel Exception occurred in display_all_bestModel", exc_info=True)
			raise e

		self.logger.info('read_DecTree_BestModel display all bestModel')

		return None




