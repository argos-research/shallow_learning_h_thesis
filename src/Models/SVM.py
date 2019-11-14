# coding=utf-8
import datetime
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
import logging
import pandas as pd
import numpy as nps
import time
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc





path = '../'

ker_str = 'kernel'
c_str = 'C'
degree_str = 'degree'
gamma_str = 'gamma'
coef_str = 'coef0'
Tol_str = 'tol'
dataset_str = 'dataset_name_path'
AcuScore_str = 'accuracy_score'
runTime_str = 'running_time'


def setup_logger(name, log_file, level=logging.INFO):
	formatter = logging.Formatter('%(asctime)s [%(filename)s: %(funcName)s - %(lineno)d] - %(message)s',
								  datefmt='%d-%b-%y %H:%M:%S)')
	handler = logging.FileHandler(log_file)
	handler.setFormatter(formatter)
	logger = logging.getLogger(name)
	logger.setLevel(level)
	logger.addHandler(handler)
	return logger


# Support vector machine
class SVM_class:
	kernel = None
	C = None
	degree = None
	gamma = None
	coef0 = None
	Tol = None

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
		{'kernel': ['linear'],
		 'degree': [1],
		 'C': [0.25]},
		{'kernel': ['linear'],
		 'C': [0.1],
		 'gamma': [1e-6, 0.25, 0.5],
		 'coef0': [0.0],
		 'tol': [0.001]},

		{'kernel': ['rbf'],
		 'C': [0.1],
		 'gamma': [1e-6, 5e-5, 1e-4],
		 'coef0': [0.0],
		 'tol': [0.001]},

		{'kernel': ['poly'],
		 'C': [0.1],
		 'degree': [2,3,4],
		 'gamma': [1e-6],
		 'coef0': [0.0, 1, 2],
		 'tol': [0.001]},
		{'kernel': ['sigmoid'],
		 'C': [0.1],
		 'gamma': [1e-6],
		 'coef0': [0.0],
		 'tol': [0.001,0.1, 0.5]},

		{'kernel': ['linear'],
		 'degree': [3],
		 'C': [0.1, 1, 50, 200, 800, 1000]},

		{'kernel': ['rbf'],
		 'gamma': [1e-3, 1e-4],
		 'C': [0.1, 1, 10, 200, 800, 1000],
		 'coef0': [0.0, 2, 3],
		 'tol': [0.001, 0.1, 1, 2.50, 10]
		 },
		{'kernel': ['poly'],
		 'C': [0.1, 1, 10, 500, 1000],
		 'gamma': [1e-6, 5e-5, 0.1, 10, 100],
		 'degree': [1, 2, 3, 5, 10, 12]},

		{'kernel': ['sigmoid'],
		 'C': [0.1, 1, 10, 500, 1000],
		 'gamma': [1e-6, 5e-5, 1e-4, 1, 10, 100],
		 'tol': [0.001, 0.1, 1, 2.50, 10],
		 'coef0': [0.0, 2, 3]}
	]


	# parameter_sets= [
	# 	{'kernel': ['linear'],
	# 	 'C': [0.1, 0.25, 0.5, 0.75, 1, 1.75, 2, 5, 10, 100, 200, 400, 600, 800, 1000],
	# 	 'gamma': [1e-6, 5e-5, 1e-4, 1e-3, 5e-3, 1e-3, 0.1, 1, 10, 100],
	# 	 'coef0': [0.0, 1, 2, 3],
	# 	 'tol': [0.001, 0.1, 0.5, 0.75, 1, 1.5, 1.75, 2, 2.50, 10]},
	#
	# 	{'kernel': ['rbf'],
	# 	 'C': [0.1, 0.25, 0.5, 0.75, 1, 1.75, 2, 5, 10, 100, 200, 400, 600, 800, 1000],
	# 	 'gamma': [1e-6, 5e-5, 1e-4, 1e-3, 5e-3, 1e-3, 0.1, 1, 10, 100],
	# 	 'coef0': [0.0, 1, 2, 3],
	# 	 'tol': [0.001, 0.1, 0.5, 0.75, 1, 1.5, 1.75, 2, 2.50, 10]},
	#
	# 	{'kernel': ['poly'],
	# 	 'C': [0.1, 0.25, 0.5, 0.75, 1, 1.75, 2, 5, 10, 100, 200, 400, 600, 800, 1000],
	# 	 'degree': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
	# 	 'gamma': [1e-6, 5e-5, 1e-4, 1e-3, 5e-3, 1e-3, 0.1, 1, 10, 100],
	# 	 'coef0': [0.0, 1, 2, 3],
	# 	 'tol': [0.001, 0.1, 0.5, 0.75, 1, 1.5, 1.75, 2, 2.50, 10]},
	#
	# 	{'kernel': ['sigmoid'],
	# 	 'C': [0.1, 0.25, 0.5, 0.75, 1, 1.75, 2, 5, 10, 100, 200, 400, 600, 800, 1000],
	# 	 'gamma': [1e-6, 5e-5, 1e-4, 1e-3, 5e-3, 1e-3, 0.1, 1, 10, 100],
	# 	 'coef0': [0.0, 1, 2, 3],
	# 	 'tol': [0.001, 0.1, 0.5, 0.75, 1, 1.5, 1.75, 2, 2.50, 10]}
	# ]


	def __init__(self, Log_BestModel, inX_data, iny_data, inXtest_data, inytest_data, infeature_name, intarget_name, dataset_name_path, kernel="linear"):
		try:
			logger = setup_logger('SVMTraining', path + 'Logging/SVMTraining.log')
			self.logger = logger
			self.dataset_name_path = dataset_name_path

			self.logger.info("\n\n\n\n")
			self.logger.info("\n\n\n\n")
			self.logger.info('=============== Initialize the SVM_class variable ==================')
			self.logger.info("===== Working on following data set: ======")
			self.logger.info(dataset_name_path)
			currentDate = " CurrentDate: " + datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
			self.logger.info(currentDate)
			self.logger.info("\n\n\n\n")

			scaling = MinMaxScaler(feature_range=(-1, 1)).fit(inX_data)
			inX_data = scaling.transform(inX_data)
			inXtest_data = scaling.transform(inXtest_data)

			# set parameter
			self.kernel = kernel
			# set data
			self.X_data = inX_data
			self.y_data = iny_data
			self.Xtest_data = inXtest_data
			self.ytest_data = inytest_data
			self.feature_names = infeature_name
			self.target_names = intarget_name
			# set model
			self.model = svm.SVC(kernel=kernel)

			self.log_BestModel_1 = Log_BestModel

			self.logger.info('Initialization Done')
			print("------------------------------ SVM_class Initialized -------------------------")
		except Exception as e:
			self.logger.error("Exception occurred in __init__", exc_info=True)
			raise e


	def generating_roc_value(self):
		try:
			# # 92.19467169066976% R3
			# in_kernel = "rbf"
			# in_tol = 0.001
			# in_gamma = 1e-06
			# in_coef0 = 0.0
			# in_C = 0.1
			# in_degree = 3
			# self.model = svm.SVC(kernel=in_kernel, tol=in_tol, C=in_C, coef0=in_coef0, gamma=in_gamma, degree=in_degree)

			# accuracy_score: accuracy_score:95.1989581094238,kernel:linear,C:0.25,degree:1,gamma:auto_deprecated,coef0:0.0,tol:0.001,
			in_kernel = "linear"
			in_tol = 0.001
			in_gamma = 'auto'
			in_coef0 = 0.0
			in_degree=1
			in_C = 0.25

			self.model = svm.SVC(kernel=in_kernel, tol=in_tol, C=in_C, coef0=in_coef0, gamma=in_gamma,degree=in_degree)
			y_score = self.model.fit(self.X_data, self.y_data).decision_function(self.Xtest_data)

			ytest=self.ytest_data
			print("\nY test")
			print(ytest)

			print("\nScore")
			print(y_score)
			fpr, tpr, thresholds = metrics.roc_curve(ytest, y_score, pos_label=None)

			roc_auc = auc(fpr, tpr)
			print(roc_auc)
			print(fpr)
			print(tpr)


			f = open("../Evaluation/SVMROC.txt", "a+")
			now = datetime.datetime.now()
			currentDate = "CurrentDate:" + now.strftime("%Y-%m-%d %H-%M")
			f.write("\n\n\n\n\n\n------------------------------ New LogRegROC Execution Started --------------------------------")
			f.write("\n---------- " + self.dataset_name_path + " -------------")
			f.write("\n---------- " + str(currentDate) + " -------------")
			f.write("\n\n------------------------------ ytest --------------------------------\n")
			f.write(str(ytest-1))
			f.write("\n\n------------------------------ y_score --------------------------------\n")
			f.write(str(y_score))
			f.write("\n\n------------------------------ roc_auc --------------------------------\n")
			f.write(str(roc_auc))
			f.write("\n\n------------------------------ fpr --------------------------------\n")
			f.write(str(fpr))
			f.write("\n\n------------------------------ tpr --------------------------------\n")
			f.write(str(tpr))
			plt.figure()
			lw = 2
			plt.plot(fpr, tpr, color='darkorange',
					 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
			plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
			plt.xlim([0.0, 1.0])
			plt.ylim([0.0, 1.05])
			plt.xlabel('False Positive Rate')
			plt.ylabel('True Positive Rate')
			plt.title('ROC')
			plt.legend(loc="lower right")

			now = datetime.datetime.now()
			currentDate = now.strftime("%Y-%m-%d_%H-%M") + "\n"
			nameOfPlot = "ROC curve"+ "_99_" + str(currentDate) + ".png"
			plt.savefig(nameOfPlot)
			nameOfPlot = "ROC curve" + "_99_" + str(currentDate) + ".svg"
			plt.savefig(nameOfPlot)

			plt.show()

			self.y_predicted = self.model.predict(self.Xtest_data)
			print("\npredictionScore")
			print(self.y_predicted)
			f.write("\npredictionScore")
			f.write(str(self.y_predicted))

			optimized_accuracy_score = metrics.accuracy_score(self.ytest_data, self.y_predicted)
			ac_score = optimized_accuracy_score * 100

			print("\nac_score")
			print(ac_score)
			f.write("\nac_score")
			f.write(str(ac_score))

		except Exception as e:
			self.logger.error("Exception occurred in train_model", exc_info=True)
			raise e
		self.logger.info('Single Model trained')
		print("------------------------------ Model trained  --------------------------------")



	def train_model(self):
		try:
				self.model.fit(self.X_data, self.y_data)
		except Exception as e:
			self.logger.error("Exception occurred in train_model", exc_info=True)
			raise e
		print("------------------------------ Model trained  ------------------------")

	def predict(self):
		try:
			self.y_predicted = self.model.predict(self.Xtest_data)
		except Exception as e:
			self.logger.error("Exception occurred in predict", exc_info=True)
			raise e
		print("------------------------------ Model predict  ------------------------")

	def calculate_accuracy_score(self):
		try:
			print("--------------------------- Accuracy score of model -------------------------------------------")
			self.accuracy_score = metrics.accuracy_score(self.ytest_data, self.y_predicted)
			print(self.accuracy_score)
			self.logger.info('Single Accuracy score of model: '+str(self.accuracy_score))
		except Exception as e:
			self.logger.error("Exception occurred in calculate_accuracy_score", exc_info=True)
			raise e

	def classification_report(self):
		try:
			print("--------------------------- classification report --------------------------------------------")
			print(metrics.classification_report(self.ytest_data, self.y_predicted, target_names=self.target_names))
		except Exception as e:
			self.logger.error("Exception occurred in classification_report", exc_info=True)
			raise e

	def confusion_matrix(self):
		try:
			print("--------------------------- confusion matrix -------------------------------------------")
			print(metrics.confusion_matrix(self.ytest_data, self.y_predicted))
		except Exception as e:
			self.logger.error("Exception occurred in confusion_matrix", exc_info=True)
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

				ker = optimized_model.best_estimator_.get_params()[ker_str]
				cri = optimized_model.best_estimator_.get_params()[c_str]
				degree = optimized_model.best_estimator_.get_params()[degree_str]
				gamma = optimized_model.best_estimator_.get_params()[gamma_str]
				coef = optimized_model.best_estimator_.get_params()[coef_str]
				tol = optimized_model.best_estimator_.get_params()[Tol_str]
				ac_score = optimized_accuracy_score * 100

				print("(\n " + AcuScore_str + ": ", ac_score
					  , ",\n " + ker_str + ": ", ker
					  , ",\n " + c_str + ": ", cri
					  ,",\n " + degree_str + ": ", degree
					  , ",\n " + gamma_str + ": ", gamma
					  , ",\n " + coef_str + ": ", coef
					  ,",\n " + Tol_str + ": ", tol
					  ,",\n " + dataset_str + ": ", self.dataset_name_path
					  , ",\n " + runTime_str + ": ", running_time, "\n)")


				parameters = AcuScore_str + ":" + str(ac_score) \
							 + "," + ker_str + ":" + str(ker) \
							 + "," + c_str + ":" + str(cri) \
							 + "," + degree_str + ":" + str(degree) \
							 + "," + gamma_str + ":" + str(gamma) \
							 + "," + coef_str + ":" + str(coef) \
							 + "," + Tol_str + ":" + str(tol) \
							 + "," + dataset_str + ":" + str(self.dataset_name_path) \
							 + "," + runTime_str + ":" + str(running_time)

				self.logger.info(parameters)
				self.log_BestModel_1.save_support_vector_machine_BestModel(parameters)

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
			bestModel_List = self.log_BestModel_1.read_support_vector_machine_BestModel()
			for index in bestModel_List:
				print(index)
				print("(\n " + AcuScore_str + ": ", bestModel_List[index][AcuScore_str],
					  ",\n " + ker_str + ": ", bestModel_List[index][ker_str],
					  ",\n " + c_str + ": ", bestModel_List[index][c_str],
					  ",\n " + degree_str + ": ", bestModel_List[index][degree_str],
					  ",\n " + gamma_str + ": ", bestModel_List[index][gamma_str],
					  ",\n " + coef_str + ": ", bestModel_List[index][coef_str],
					  ",\n " + Tol_str + ": ", bestModel_List[index][Tol_str],
					  ",\n " + dataset_str + ": ", bestModel_List[index][dataset_str],
					  ",\n " + runTime_str + ": ", bestModel_List[index][runTime_str],
					  ",\n CurrentDate: ", bestModel_List[index]['CurrentDate'],
					  "\n)")

		except Exception as e:
			self.logger.error("Exception occurred in display_all_bestModel", exc_info=True)
			raise e

		self.logger.info('display all bestModel')

		return None