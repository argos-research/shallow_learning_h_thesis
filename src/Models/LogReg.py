import datetime
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import logging
import numpy as np
import pandas as pd
import time

path = '../'
# parameter string
pen_str = 'penalty'
dual_str = 'dual'
solv_str = 'solver'
c_str = 'C'
mClass_str = 'multi_class'
maxIt_str = 'max_iter'
Tol_str = 'tol'
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

path = '../'

# Logistic regression
class LogReg_class:
	penalty = None
	tol = None
	C = None
	solver = None
	max_iter = None
	multi_class = None
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
		# 1
		{
			'penalty': ['l2'],
			 'dual': [True],
			 'solver': ['liblinear'],
			 'C': [100],
			 'multi_class': ['ovr'],
			 'max_iter': [110],
			 'tol': [1e-4]
		 },
		# 2
		{
			'penalty': ['l1'],
			'solver': ['saga'],
			'C': [0.001, 1, 10, 100, 1000],
			'max_iter': [50, 100, 150],
			'multi_class': ['ovr', 'multinomial'],
			'tol': [1e-4]
		},
		# 3
		{
			'penalty': ['l2'],
			'solver': ['sag'],
			'C': [0.01, 0.1, 1, 10],
			'multi_class': ['ovr', 'multinomial'],
			'max_iter': [100, 150],
			'tol': [1e-4]
		},
		# 4
		{
			'penalty': ['l2'],
			'solver': ['lbfgs'],
			'C': [0.001, 2.0, 10, 100, 1000],
			'multi_class': ['ovr', 'multinomial'],
			'max_iter': [100, 140, 200],
			'tol': [1e-4]
		},
		# 5
		{
			'penalty': ['l2'],
			'solver': ['newton-cg'],
			'C': [0.001, 1, 2.5, 100, 1000],
			'multi_class': ['ovr', 'multinomial'],
			'max_iter': [100, 150, 200],
			'tol': [1e-4]
		},
		# 6
		{   'penalty': ['l1','l2'],
			'solver': ['liblinear'],
			'C': np.logspace(-4, 4, 5),
			'multi_class': ['ovr'],
			'max_iter': [100,130,140],
		},
		{
			'penalty': ['elasticnet'],
			'solver': ['saga'],
			'C': [0.001, 1, 10, 100, 1000],
			'max_iter': [50, 100, 150],
			'multi_class': ['ovr'],
			'tol': [1e-4],
			'l1_ratio': [0, 0.5, 0.2, 0.75, 1]
		},
		{
			'penalty': ['elasticnet'],
			'solver': ['saga'],
			'C': [0.001, 1, 10, 100, 1000],
			'max_iter': [50, 100, 150],
			'multi_class': ['multinomial'],
			'tol': [1e-4],
			'l1_ratio': [0, 0.5, 0.2, 0.75, 1]
		},

	]


	def __init__(self, Log_BestModel, inX_data, iny_data, inXtest_data, inytest_data, infeature_name, intarget_name, dataset_name_path, in_penalty='l2', in_tol=1e-4, in_C=1.0, in_solver='liblinear', in_max_iter=1000, in_multi_class='ovr'):
		try:
			logger = setup_logger('LogRegTraining', path + 'Logging/LogRegTraining.log')
			self.logger = logger
			self.dataset_name_path = dataset_name_path

			self.logger.info("\n\n\n\n")
			self.logger.info("\n\n\n\n")
			self.logger.info('=============== Initialize the LogReg_class variable ==================')
			self.logger.info("===== Working on following data set: ======")
			self.logger.info(dataset_name_path)
			currentDate = " CurrentDate: " + datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
			self.logger.info(currentDate)
			self.logger.info("\n\n\n\n")

			scaling = MinMaxScaler(feature_range=(-1, 1)).fit(inX_data)
			inX_data = scaling.transform(inX_data)
			inXtest_data = scaling.transform(inXtest_data)


			# set parameter
			self.penalty = in_penalty
			self.tol = in_tol
			self.C = in_C
			self.solver = in_solver
			self.max_iter = in_max_iter
			self.multi_class = in_multi_class

			# set data
			self.X_data = inX_data
			self.y_data = iny_data
			self.Xtest_data = inXtest_data
			self.ytest_data = inytest_data
			self.feature_names = infeature_name
			self.target_names = intarget_name

			# set model
			self.model = linear_model.LogisticRegression(penalty=in_penalty, tol=in_tol, C=in_C, solver=in_solver, max_iter=in_max_iter, multi_class=in_multi_class)
			self.log_BestModel_1 = Log_BestModel

			print("------------------------------ LogReg_class Initialized ----------------------")
		except Exception as e:
			self.logger.error("Exception occurred in __init__", exc_info=True)
			raise e


	def generating_roc_value(self):
		try:

			in_penalty = 'elasticnet'
			in_dual = False
			in_solver = 'saga'
			in_tol = 0.0001
			in_C = 10
			in_max_iter = 50
			in_multi_class = 'multinomial'
			in_l1_ratio = 0

			self.model = linear_model.LogisticRegression(penalty=in_penalty, tol=in_tol, C=in_C, solver=in_solver,max_iter=in_max_iter, multi_class=in_multi_class,l1_ratio=in_l1_ratio,dual=in_dual )

			# calculate score
			y_score = self.model.fit(self.X_data, self.y_data).decision_function(self.Xtest_data)

			ytest=self.ytest_data
			# ytest += 1
			print("\nY test")
			print(ytest)

			print("\nScore")
			print(y_score)
			fpr, tpr, thresholds = metrics.roc_curve(ytest, y_score, pos_label=None)

			roc_auc = auc(fpr, tpr)
			print(roc_auc)
			print(fpr)
			print(tpr)

			f = open("../Evaluation/LogRegROC.txt", "a+")
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
			plt.title('ROC for Logistic Regression on Dataset1')
			plt.legend(loc="lower right")

			now = datetime.datetime.now()
			currentDate = now.strftime("%Y-%m-%d_%H-%M") + "\n"
			nameOfPlot = "ROC curve_R3"+ "_95.4_" + str(currentDate) + ".png"
			plt.savefig(nameOfPlot)
			nameOfPlot = "ROC curve_R3" + "_95.4_" + str(currentDate) + ".svg"
			plt.savefig(nameOfPlot)

			plt.show()

			# self.ytest_data-=1
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
			# self.classification_report()
			# self.confusion_matrix()

		except Exception as e:
			self.logger.error("Exception occurred in train_model", exc_info=True)
			raise e
		self.logger.info('Single Model trained')
		print("------------------------------ Model trained  --------------------------------")


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
				self.logger.info('------------------------- Random-Search improving Pv6 results on R3------------------------')

			for hyperparameters in self.parameter_sets:

				i += 1
				currentDate = "    CurrentDate: " + datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
				searchIteration = "    Search iteration: "+ str(i)
				self.logger.info("\n\n")

				self.logger.info('------------------------- Hyperparameoptimized_accuracy_score = metrics.accuracy_score(self.ytest_data, self.optimized_y_predicted)ter_optimization_new ------------------------')
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

				pen_value = optimized_model.best_estimator_.get_params()[pen_str]
				dual_value = optimized_model.best_estimator_.get_params()[dual_str]
				solv_value = optimized_model.best_estimator_.get_params()[solv_str]
				c_value = optimized_model.best_estimator_.get_params()[c_str]
				mClass_value = optimized_model.best_estimator_.get_params()[mClass_str]
				maxIt_value = optimized_model.best_estimator_.get_params()[maxIt_str]
				Tol_value = optimized_model.best_estimator_.get_params()[Tol_str]
				ac_score = optimized_accuracy_score * 100

				print("(\n " + AcuScore_str + ": ", ac_score,
					  ",\n " + pen_str + ": ", pen_value,
					  ",\n " + dual_str + ": ", dual_value,
					  ",\n " + solv_str + ": ", solv_value,
					  ",\n " + c_str + ": ", c_value,
					  ",\n " + mClass_str + ": ", mClass_value,
					  ",\n " + maxIt_str + ": ", maxIt_value,
					  ",\n " + Tol_str + ": ", Tol_value,
					  ",\n " + dataset_str + ": ", self.dataset_name_path,
					  ",\n " + runTime_str + ": ", running_time, "\n)")

				parameters = AcuScore_str + ":" + str(ac_score) \
							 + "," + pen_str + ":" + str(pen_value) \
							 + "," + dual_str + ":" + str(dual_value)\
							 + "," + solv_str + ":" + str(solv_value)\
							 + "," + c_str + ":" + str(c_value)\
							 +"," + mClass_str + ":" + str(mClass_value)\
							 +"," + maxIt_str + ":" + str(maxIt_value)\
							 +"," + Tol_str + ":" + str(Tol_value)\
							 +"," + dataset_str + ":" + str(self.dataset_name_path)\
							 + "," + runTime_str + ":" + str(running_time)






				self.logger.info(parameters)
				self.log_BestModel_1.save_logReg_BestModel(parameters)

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
			bestModel_List = self.log_BestModel_1.read_logReg_BestModel()
			for index in bestModel_List:
				print(index)
				print("(\n " + AcuScore_str + ": ", bestModel_List[index][AcuScore_str],
					  ",\n " + pen_str + ": ", bestModel_List[index][pen_str],
					  ",\n " + dual_str + ": ", bestModel_List[index][dual_str],
					  ",\n " + solv_str + ": ", bestModel_List[index][solv_str],
					  ",\n " + c_str + ": ", bestModel_List[index][c_str],
					  ",\n " + mClass_str + ": ", bestModel_List[index][mClass_str],
					  ",\n " + maxIt_str + ": ", bestModel_List[index][maxIt_str],
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
