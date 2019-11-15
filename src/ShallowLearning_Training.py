# -*- coding: utf-8 -*-
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from src.Log_BestModel import Log_BestModel
from src.Models.RanFor import Ranfor_class
from src.Models.SVM import SVM_class
from src.Models.KNN import KNN_class
from src.Models.LogReg import LogReg_class
from src.Models.DecTree import DecTree_class
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1   - Data Collection
# 2   - Data Preparation
# 2.1 - Read data from db
# 2.4 - Feature Scaling
# 2.5 - Selecting Meaningful Features
# 2.6 - Save Data in pickle file
# 2.7 - Load Data from pickle file
# 2.8 - Splitting Data Into Subsets
# 3   - Choose a Model
# 4   - Train the Model
# 5   - Evaluate the Model
# 6   - Parameter Tuning
# 7   - Make Predictions

path = '../'

# name of all the directory wher the feature set is contained
DataSetV3 = 'Datasets/FinalDatasetV3/'
DataSetV5 = 'Datasets/FinalDatasetV5/'
DataSetV6 = 'Datasets/FinalDatasetV6/'
DataSetBothV6 = 'Datasets/FinalDatasetBothV6/'
DataSetR3 = 'Datasets/FinalDatasetR3/'
TestDataset = 'Datasets/TestDataset/'

# helping function for calculating the execution time of each task
def startTimer():
	return time.time()

# helping function for calculating the execution time of each task
def endTimer(message,sTime):
	print(message+" execution time")
	print("--- %s seconds ---" % (time.time() - sTime))
	return



def graph_plotting_functions(xTrain, yTrain):

	# saving individual features in file
	np.savetxt(r'Plotting/1CriticalTime1.txt', xTrain['CriticalTime1'].head(400), fmt='%d')
	np.savetxt(r'Plotting/1Priority1.txt', xTrain['Priority1'].head(400), fmt='%d')
	np.savetxt(r'Plotting/1AvgT1.txt', xTrain['AvgT1'].head(400), fmt='%d')

	# scatter plot for two features using matplotlib.pyplot
	x = xTrain['CriticalTime1'].head(1000)
	y = xTrain['AvgT1'].head(1000)

	setosa_x = x[:50]
	setosa_y = y[:50]

	versicolor_x = x[50:]
	versicolor_y = y[50:]

	plt.figure(figsize=(8, 6))
	plt.scatter(setosa_x, setosa_y, marker='+', color='green')
	plt.scatter(versicolor_x, versicolor_y, marker='_', color='red')
	plt.show()

	# scatter plot for two features using matplotlib.pyplot
	plt.scatter(xTrain['AvgT1'].head(500), xTrain['Priority1'].head(500), alpha=0.2,
				s=500, c=yTrain.head(500), cmap='viridis')
	plt.xlabel("CriticalTime1")
	plt.ylabel("Priority1")
	plt.show()

	# scatter plot for CriticalTime1 x Priority1 features using seaborn _ saving the plot
	ax = sns.scatterplot(x="CriticalTime1", y="Priority1", hue="PKG1", data=xTrain.head(1000))
	ax.get_figure().savefig("Plotting/1Priority1xCriticalTime1xPKG1.png")

	# scatter plot for CriticalTime1 x  AvgT1 features using seaborn _ saving the plot
	ax = sns.scatterplot(x="CriticalTime1", y="AvgT1", data=xTrain.head(5000))
	ax.get_figure().savefig("Plotting/1CriticalTime1xAvgT1.png")

	# scatter plot for CriticalTime1 x  AvgT1 features and grouped on PKG1 using seaborn _ saving the plot
	ax = sns.scatterplot(x="CriticalTime1", y="AvgT1",hue="PKG1", data=xTrain.head(3000))
	ax.get_figure().savefig("Plotting/1CriticalTime1xAvgT1xPKG1.png")

	return None

# load the dataset from the datasets data
def load_dataset(DataSetPath):
	try:
		logging.info("Load_data")
		logging.info("reading pickle data")

		df_feature_data = pd.read_pickle(path + DataSetPath + "feature_data.pkl")
		df_target_data = pd.read_pickle(path + DataSetPath + "target_data.pkl")

		df_feature_names = pd.read_pickle(path + DataSetPath + "feature_names.pkl")
		# df_target_names = pd.read_pickle(path + "CleanData/3_target_names.pkl")
		df_target_names = target_names = ['class_NotExec_0', 'class_Exec_1']
		print("------------------------------ Data loaded -----------------------------------")
		return df_feature_data, df_target_data, df_feature_names, df_target_names
	except Exception as e:
		logging.info('Error in data loading')
		logging.error("Exception occurred in Load_data", exc_info=True)
		raise e

	return None

# split the dataset into test and train
def split_dataset(x, y):
	try:
		logging.info("splitting dataset via sklearn train_test_split function")
		# split with ratio of 80 and 20
		xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=0)
		return xTrain, xTest, yTrain, yTest
	except Exception as e:
		logging.info('Error in splitting data')
		logging.error("Exception occurred in split_dataset", exc_info=True)
		raise e

	return None

# train a single model
def run_model(SL_algo):
	try:
		s = startTimer()

		# train single model
		SL_algo.train_model()
		endTimer("train_model ", s)

		s = startTimer()

		# predict via trained model
		SL_algo.predict()
		endTimer("predict ", s)
		s = startTimer()

		# calculate accuracy score
		SL_algo.calculate_accuracy_score()


		endTimer("calculate_accuracy_score ", s)
		s = startTimer()

		SL_algo.classification_report()

		endTimer("classification_report ", s)
		s = startTimer()

		# display the confusion matrix
		SL_algo.confusion_matrix()
		endTimer("confusion_matrix ", s)

	except Exception as e:
		logging.info('Error in run_model')
		logging.error("Exception occurred in run_model", exc_info=True)
		raise e



# hyper parameter optimization Call
# SL_algo.graph()
# SL_algo.hyperparameter_optimization_search()
# SL_algo.training_model_n_feature_importance()

# function to run hyper parameter search, it calls the shallow learning class function
def run_bestModel_search(SL_algo):
	try:
		s = startTimer()
		# by default grid search is enabled
		SL_algo.hyperparameter_optimization_search()
		endTimer("hyperparameter_optimization_search ", s)
	except Exception as e:
		logging.info('Error in run_bestModel_search')
		logging.error("Exception occurred in run_bestModel_search", exc_info=True)
		raise e

# function to display all the best models
def display_trainedModel_results(SL_algo):
	try:
		s = startTimer()
		SL_algo.display_all_bestModel()
		endTimer("display_all_bestModel() ", s)
	except Exception as e:
		logging.info('Error in display_trainedModel_results')
		logging.error("Exception occurred in display_trainedModel_results", exc_info=True)
		raise e


# Command line interface functions for selecting the database for training
def select_dataset():
	print('==============================================================================')
	print('-------------------------  Select the dataset  -------------------------------')
	print('==============================================================================\n')
	print("Select the Dataset:")
	print("type '1' for 'rpi3_final.db' Database")
	print("type '2' for 'panda_v6both.db' Database")
	print("type '3' for 'TestDataset' Database")
	db_option = input("Enter input: ")
	if(db_option == '1'):
		return DataSetR3
	elif(db_option == '2'):
		return DataSetBothV6
	else:
		return TestDataset
	return TestDataset




# command line function for selecting shallow learning algorithm - returns the variable for shallow learning class
def select_shallowLearning_algorithm(log_BestModel_1, xTrain, yTrain, xTest, yTest, df_feature_names, df_target_names, dataset_name):
	print('\n\n==========================================================================')
	print('-------------------------  Select the Shallow learning algorithm  ------------')
	print('==============================================================================\n')
	print("Select the Shallow Learning algorithm for training:")
	print("type '1' for Random Forest")
	print("type '2' for Support Vector Machine")
	print("type '3' for Logistic Regression")
	print("type '4' for K nearest Neighbor")
	print("type '5' for Decision Tree")

	db_option = input("Enter input: ")

	# one of the shallow learning algorithm variable is created and returned.
	if(db_option == '1'):
		return Ranfor_class(log_BestModel_1, xTrain, yTrain, xTest, yTest, df_feature_names, df_target_names,dataset_name)
	elif(db_option == '2'):
		return SVM_class(log_BestModel_1, xTrain, yTrain, xTest, yTest, df_feature_names, df_target_names, dataset_name)
	elif(db_option == '3'):
		return  LogReg_class(log_BestModel_1, xTrain, yTrain, xTest, yTest, df_feature_names, df_target_names, dataset_name)
	elif(db_option == '4'):
		return KNN_class(log_BestModel_1, xTrain, yTrain, xTest, yTest, df_feature_names, df_target_names, dataset_name)
	elif(db_option == '5'):
		return DecTree_class(log_BestModel_1, xTrain, yTrain, xTest, yTest, df_feature_names, df_target_names, dataset_name)

	# by default random class is always returned.
	return Ranfor_class(log_BestModel_1, xTrain, yTrain, xTest, yTest, df_feature_names, df_target_names,dataset_name)


# command line function for selecting the training type
def select_type_of_training():
	print('\n\n==========================================================================')
	print('-------------------------  Select Training Type ------------------------------')
	print('==============================================================================\n')
	print("Select the training type:")
	print("type '1' for One model training")
	print("type '2' for Hyperparameter optimization")
	db_option = input("Enter input: ")
	if(db_option == '1'):
		return 1
	elif(db_option == '2'):
		return 2
	return 2



def main_program():
	try:
		#select dataset
		dataset_name_path = select_dataset()

		# Load Dataset
		df_feature_data, df_target_data, df_feature_names, df_target_names = load_dataset(dataset_name_path) # feature_data, target_data, feature_names, target_names

		# split dataset into training and test data
		xTrain, xTest, yTrain, yTest = split_dataset(df_feature_data, df_target_data)

		# initialize logger
		log_BestModel_1 = Log_BestModel()

		# select and initialize shallowlearning algorithm
		algorithm_class = select_shallowLearning_algorithm(log_BestModel_1, xTrain, yTrain, xTest, yTest, df_feature_names, df_target_names, dataset_name_path+"feature_data.csv")

		# select the type of training to perform, optimization or single model training
		training_type = select_type_of_training()

		if (training_type == 1):
			# run single model training
			run_model(algorithm_class)
		elif (training_type == 2):
			# run hyper parameter search training
			run_bestModel_search(algorithm_class)

		# Display Trained Results
		display_trainedModel_results(algorithm_class)

	except Exception as e:
		logging.info('Error while processing the main_program operations')
		logging.error("Exception occurred in main_program", exc_info=True)
		raise e

	return


if __name__ == "__main__":
	logging.basicConfig(filename=path + 'Logging/Training.log',
						format='%(asctime)s [%(filename)s: %(funcName)s - %(lineno)d] - %(message)s',
						datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)
	logging.info('main_program start')

	try:
		#starting the shallow learning model building
		main_program()
	except Exception as e:
		logging.info('Error in main program execution')
		logging.error("Exception occurred in mainProgram", exc_info=True)
		raise e


