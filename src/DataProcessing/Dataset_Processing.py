import sqlite3
from sqlite3 import Error
import pandas as pd
import logging
import time


# -----------------------------
path = '../../'
start_time = 0

# Databases name variable
dbnamer3 = 'rpi3_final.db'
dbnamev6 = 'panda_v6both.db'
# dbnewDatabase = ''

# Database - dataset saving path/directory reference
datasetDirectory_path_r3 = "FinalDatasetR3/"
datasetDirectory_path_pv6 = "FinalDatasetBothV6/"
# datasetDirectory_path_newdb = ""

selected_db = "-"
saveFile_name = "-"
sql_job_table = ''
sql_select_job_table = ''

sql_task1_table = ''
sql_select_task1_table = ''

sql_task2_table = ''
sql_select_task2_table = ''

sql_task3_table = ''
sql_select_task3_table = ''

sql_taskset_table = ''
sql_select_taskset_table = ''


# ----------------------------------------------------
# Data transformation arrays list of columns to drop
# ----------------------------------------------------

# 1
# constant values:  not changing for each task
constant_values = ["Deadline", "Quota", "Caps", "CORES", "COREOFFSET", "Offset"]

# 2
# IDs: not useful in training
Ids_task1 = ["Set_ID", "TASK1_ID"]
Ids_task2 = ["Set_ID", "TASK1_ID", "TASK2_ID"]
Ids_task3 = ["Set_ID", "TASK1_ID", "TASK2_ID", "TASK3_ID"]
# All_IDs_taskset = ["Set_ID", "TASK1_ID", "TASK2_ID", "TASK3_ID", "Set_ID:1", "TASK1_ID:1","Set_ID:2", "TASK1_ID"]

# All_IDs_taskset = ["Unnamed: 0","Set_ID", "TASK1_ID", "TASK2_ID", "TASK3_ID","TASK4_ID"
# 					,"Set_ID:1", "TASK1_ID:1", "Successful:1"
# 					,"Set_ID:2", "TASK2_ID:1", "Successful:2"
# 					,"Set_ID:3", "TASK3_ID:1", "Successful:3"]

All_IDs_taskset = ["Set_ID", "TASK1_ID", "TASK2_ID", "TASK3_ID","TASK4_ID"
					,"Set_ID:1", "TASK1_ID:1", "Successful:1"
					,"Set_ID:2", "TASK2_ID:1", "Successful:2"
					,"Set_ID:3", "TASK3_ID:1", "Successful:3"]

# -----------------------------
# string to numeric conversions
# -----------------------------

# 1
# Successful from table of task already in numeric form
# 0 not_executable
# 1 executable

# 2
# an interger is assign to each PKG text label from Task table
pkg_values = { 'pi': 0,
                'hey': 1,
                'tumatmul': 2,
                'cond_mod': 3
             }

# 3
# arg value is scaled between range from 1 to 17
arg_values = { '1' : 1,
                '4096' : 2,
                '8192' : 3,
                '16384' : 4,
                '32768' : 5,
                '65536' : 6,
                '131072' : 7,
                '262144' : 8,
                '524288' : 9,
                '1048576' : 10,
                '2097152' : 11,
                '847288609443' : 12,
                '2541865828329' : 13,
                '7625597484987' : 14,
                '22876792454961' : 15,
                '68630377364883' : 16,
                '205891132094649' :17,
				'1.0' : 1,
                '4096.0' : 2,
                '8192.0' : 3,
                '16384.0' : 4,
                '32768.0' : 5,
                '65536.0' : 6,
                '131072.0' : 7,
                '262144.0' : 8,
                '524288.0' : 9,
                '1048576.0' : 10,
                '2097152.0' : 11,
                '847288609443.0' : 12,
                '2541865828329.0' : 13,
                '7625597484987.0' : 14,
                '22876792454961.0' : 15,
                '68630377364883.0' : 16,
                '205891132094649.0' :17
              }

# integer value is assigned to both Exit Values from Job table
exit_values = { 'EXIT_CRITICAL':0,
                'EXIT':1
              }

# integer value is assigned to Successful column from Taskset table
Successful = {'NOTEXECUTABLE': [0],
              'EXECUTABLE': [1]
              }






# panda_v6both SQL Quries
sql_job_table_PV6 = ('CREATE TEMPORARY TABLE JOB_INFO AS '
				 'SELECT '
				 'Set_ID '
				 ',Task_ID '
				 ',Avg(Job.End_Date - Job.Start_Date)  as AvgTime '
				 ',min(Job.End_Date - Job.Start_Date) as MinTime '
				 ',Max(Job.End_Date - Job.Start_Date) as MaxTime '
				 'FROM '
				 'Job '
				 'GROUP BY Set_ID, Task_ID'
				 )

sql_select_job_table_PV6 = ('SELECT * '
						'FROM JOB_INFO')

sql_task1_table_PV6 = ('CREATE TEMPORARY TABLE Task1_INFO AS '
					   'SELECT TaskSet.Set_ID '
					   ',TaskSet.TASK1_ID '
					   ',TaskSet.Successful '
					   ',TASK.Priority as Priority1 '
					   ',Task.PKG as PKG1 '
					   ',Task.Arg as Arg1 '
					   ',(Task.Period/1000) as Period1 '
					   ',(Task.CriticalTime/1000) as CriticalTime1 '
					   ',Task.Number_of_Jobs as JobCount1 '
					   ',Task.MAX_RUNTIME as MaxT1 '
					   ',Task.MIN_RUNTIME as MinT1 '
					   ',CASE WHEN AVG_RUNTIME < -1 THEN ROUND((AVG_RUNTIME*-1), 2) '
					   'WHEN AVG_RUNTIME >-1 THEN ROUND(AVG_RUNTIME, 2) '
					   'ELSE AVG_RUNTIME '
					   'END AS AvgT1 '
					   'FROM '
					   'TaskSet '
					   'LEFT JOIN '
					   'TASK '
					   'ON '
					   'TASKSET.TASK1_ID = TASK.Task_ID'
					   )

sql_select_task1_table_PV6 = ('SELECT * '
						'FROM Task1_INFO')

sql_task2_table_PV6 = ('CREATE TEMPORARY TABLE Task2_INFO AS '
					   'SELECT '
					   'TaskSet.Set_ID '
					   ',TaskSet.TASK2_ID '
					   ',TaskSet.Successful '
					   ',TASK.Priority as Priority2 '
					   ',Task.PKG as PKG2 '
					   ',Task.Arg as Arg2 '
					   ',(Task.Period/1000) as Period2 '
					   ',(Task.CriticalTime/1000) as CriticalTime2 '
					   ',Task.Number_of_Jobs as JobCount2 '
					   ',Task.MAX_RUNTIME as MaxT2 '
					   ',Task.MIN_RUNTIME as MinT2 '
					   ',CASE WHEN AVG_RUNTIME < -1 THEN ROUND((AVG_RUNTIME*-1), 2) '
					   'WHEN AVG_RUNTIME >-1 THEN ROUND(AVG_RUNTIME, 2) '
					   'ELSE AVG_RUNTIME '
					   'END AS AvgT2 '
					   'FROM '
					   'TaskSet '
					   'LEFT JOIN '
					   'TASK '
					   'ON '
					   'TASKSET.TASK2_ID = TASK.Task_ID;')

sql_select_task2_table_PV6 = ('SELECT * '
						'FROM Task2_INFO')



sql_task3_table_PV6 = ('CREATE TEMPORARY TABLE Task3_INFO AS '
					   'SELECT '
					   'TaskSet.Set_ID '
					   ',TaskSet.TASK3_ID '
					   ',TaskSet.Successful '
					   ',TASK.Priority as Priority3 '
					   ',Task.PKG as PKG3 '
					   ',Task.Arg as Arg3 '
					   ',(Task.Period/1000) as Period3 '
					   ',(Task.CriticalTime/1000) as CriticalTime3 '
					   ',Task.Number_of_Jobs as JobCount3 '
					   ',Task.MAX_RUNTIME as MaxT3 '
					   ',Task.MIN_RUNTIME as MinT3 '
					   ',CASE WHEN AVG_RUNTIME < -1 THEN ROUND((AVG_RUNTIME*-1), 2) '
					   'WHEN AVG_RUNTIME >-1 THEN ROUND(AVG_RUNTIME, 2) '
					   'ELSE AVG_RUNTIME '
					   'END AS AvgT3 '
					   'FROM '
					   'TaskSet '
					   'LEFT JOIN '
					   'TASK '
					   'ON '
					   'TASKSET.TASK3_ID = TASK.Task_ID'
					   )

sql_select_task3_table_PV6 = ('SELECT * '
						'FROM Task3_INFO')


# Raspberry pi 3 SQL Quries
sql_job_table_R3 = ('CREATE TEMPORARY TABLE JOB_INFO AS '
				 'SELECT '
				 'Set_ID '
				 ',Task_ID '
				 ',Round(Avg(TimeDifference2),2)  as AvgTime '
				 ',min(TimeDifference2) as MinTime '
				 ',Max(TimeDifference2) as MaxTime '
				 'FROM ' 
				 '( '
				 '	SELECT '
				 '	Job.Set_ID '
				 '	,Job.Task_ID '
				 '	,(Job.End_Date - Job.Start_Date)  as TimeDifference1 '
				 '	,CASE WHEN Job.End_Date  < Job.Start_Date THEN (4294967 - Job.Start_Date)+Job.End_Date '
				 '	WHEN Job.End_Date > Job.Start_Date THEN Job.End_Date - Job.Start_Date '
				 '	ELSE (Job.End_Date - Job.Start_Date) '
				 '	END AS TimeDifference2 '
				 '	FROM Job '
				 ') '
				 'GROUP BY Set_ID, Task_ID'
				 )



sql_select_job_table_R3 = ('SELECT * '
						'FROM JOB_INFO')

sql_task1_table_R3 = ('CREATE TEMPORARY TABLE Task1_INFO AS '
				   'SELECT TaskSet.Set_ID '
				   ',TaskSet.TASK1_ID '
				   ',TaskSet.Successful '
				   ',TASK.Priority as Priority1 '
				   ',Task.PKG as PKG1 '
				   ',Task.Arg as Arg1 '
				   ',(Task.Period/1000) as Period1 '
				   ',(Task.CriticalTime/1000) as CriticalTime1 '
				   ',Task.Number_of_Jobs as JobCount1 '
				   ',jo.AvgTime as AvgT1 '
				   ',jo.MaxTime as MaxT1 '
				   ',jo.MinTime as MinT1 '
				   'FROM '
				   'TaskSet '
				   'LEFT JOIN '
				   'TASK '
				   'ON '
				   'TASKSET.TASK1_ID = TASK.Task_ID '
				   'LEFT JOIN '
				   'JOB_INFO as jo '
				   'on '
				   'jo.Set_ID = TaskSet.Set_ID '
				   'AND '
				   'jo.Task_ID = TaskSet.TASK1_ID'
				   )

sql_select_task1_table_R3 = ('SELECT * '
						'FROM Task1_INFO')




sql_task2_table_R3 = ('CREATE TEMPORARY TABLE Task2_INFO AS '
				   'SELECT '
				   'TaskSet.Set_ID '
				   ',TaskSet.TASK2_ID '
				   ',TaskSet.Successful '
				   ',TASK.Priority as Priority2 '
				   ',Task.PKG as PKG2 '
				   ',Task.Arg as Arg2 '
				   ',(Task.Period/1000) as Period2 '
				   ',(Task.CriticalTime/1000) as CriticalTime2 '
				   ',Task.Number_of_Jobs as JobCount2 '
				   ',jo.AvgTime as AvgT2 '
				   ',jo.MaxTime as MaxT2 '
				   ',jo.MinTime as MinT2 '
				   'FROM '
				   'TaskSet '
				   'LEFT JOIN '
				   'TASK '
				   'ON '
				   'TASKSET.TASK2_ID = TASK.Task_ID '
				   'LEFT JOIN '
				   'JOB_INFO as jo '
				   'on '
				   'jo.Set_ID = TaskSet.Set_ID '
				   'AND '
				   'jo.Task_ID = TaskSet.TASK2_ID')

sql_select_task2_table_R3 = ('SELECT * '
						'FROM Task2_INFO')



sql_task3_table_R3 = ('CREATE TEMPORARY TABLE Task3_INFO AS '
				   'SELECT '
				   'TaskSet.Set_ID '
				   ',TaskSet.TASK3_ID '
				   ',TaskSet.Successful '
				   ',TASK.Priority as Priority3 '
				   ',Task.PKG as PKG3 '
				   ',Task.Arg as Arg3 '
				   ',(Task.Period/1000) as Period3 '
				   ',(Task.CriticalTime/1000) as CriticalTime3 '
				   ',Task.Number_of_Jobs as JobCount3 '
				   ',jo.AvgTime as AvgT3 '
				   ',jo.MaxTime as MaxT3 '
				   ',jo.MinTime as MinT3 '
				   'FROM '
				   'TaskSet '
				   'LEFT JOIN '
				   'TASK '
				   'ON '
				   'TASKSET.TASK3_ID = TASK.Task_ID '
				   'LEFT JOIN '
				   'JOB_INFO as jo '
				   'on '
				   'jo.Set_ID = TaskSet.Set_ID '
				   'AND '
				   'jo.Task_ID = TaskSet.TASK3_ID')

sql_select_task3_table_R3 = ('SELECT * '
						'FROM Task3_INFO')





# Taskset query is same for both databases
sql_taskset_table = ('CREATE TEMPORARY TABLE Taskset_INFO AS '
					'select * '
					'from TaskSet '
					'LEFT JOIN Task1_INFO '
					'ON '
					'TaskSet.Set_ID = Task1_INFO.Set_ID '
					'LEFT JOIN Task2_INFO '
					'ON '
					'TaskSet.Set_ID = Task2_INFO.Set_ID '
					'LEFT JOIN Task3_INFO '
					'ON '
					'TaskSet.Set_ID = Task3_INFO.Set_ID'
					)

sql_select_taskset_table = ('select * '
						   'from Taskset_INFO'
						    )

def startTimer():
	return time.time()

def endTimer(message,sTime):
	print(message+" reading time")
	print("--- %s seconds ---" % (time.time() - sTime))
	return

# Functions for generating the dataset as csv Files

def create_db_connection(db_name):
    try:
        db = sqlite3.connect(db_name)
        logging.info("db connection created Successful")
        return db
    except Error as e:
        print(e)
        logging.error("Exception occurred in create_db_connection", exc_info=True)
    return None

def close_db_connection(db_name):
    try:
        db.close()
        logging.info("db connection closed Successful")
    except Error as e:
        print(e)
        logging.error("Exception occurred in closed_db_connection", exc_info=True)
    return None

def read_all_jobs(db):
	try:
		s = startTimer()
		c = db.cursor()
		c.execute(sql_job_table)
		df_jobs = pd.read_sql(sql_select_job_table, db)
		endTimer("Job Table", s)
		logging.info("Jobs Table read successfully")
	except Exception as e:
		logging.info('Jobs Table not readable')
		db.rollback() # Roll back any change if something goes wrong
		logging.error("Exception occurred in read_all_jobs", exc_info=True)
		raise e
	finally:
		return	df_jobs
	return None

def read_all_tasks(db):
	try:
		s = startTimer()
		c = db.cursor()
		c.execute(sql_task1_table)
		c.execute(sql_task2_table)
		c.execute(sql_task3_table)

		df_task1 = pd.read_sql(sql_select_task1_table, db)
		df_task2 = pd.read_sql(sql_select_task2_table, db)
		df_task3 = pd.read_sql(sql_select_task3_table, db)

		endTimer("Tasks Table", s)
		logging.info("Tasks Table read successfully")
	except Exception as e:
		logging.info('Tasks Table not readable')
		db.rollback() # Roll back any change if something goes wrong
		logging.error("Exception occurred in read_all_tasks", exc_info=True)
		raise e
	finally:
		return	df_task1

	return None

def read_all_tasksets(db):
	try:
		s = startTimer()
		c = db.cursor()
		c.execute(sql_taskset_table)
		df_tasksets = pd.read_sql(sql_select_taskset_table, db)
		endTimer("Tasksets Table", s)
		logging.info("Tasksets Table read successfully")
	except Exception as e:
		logging.info('Tasksets Table not readable')
		db.rollback() # Roll back any change if something goes wrong
		logging.error("Exception occurred in read_all_tasksets", exc_info=True)
		raise e
	finally:
		return	df_tasksets

	return None

def read_all_data(db):
	try:
		s = startTimer()
		# read all job information and calculating avg min max time for rasp 3
		df_jobs = read_all_jobs(db)

		# read all the information of tasks and concatinate them with job information
		df_tasks = read_all_tasks(db)

		# place information of all the tasks side by side in taskset array - generate final taskset array
		df_tasksets = read_all_tasksets(db)

		endTimer("\n reading time for processing all Tasksets and its values", s)
		logging.info("read_all_data function read successful")
	except Exception as e:
		logging.error("Exception occurred in read_all_data", exc_info=True)
		raise e
	finally:
		logging.info('return Taskset with all values')
		return  df_tasksets

	return None

def save_tasksets_data(df_tasksets, FileName):
    try:
        logging.info("-----------------saving data-----------------------")
        df_tasksets.to_csv(path + "Datasets/"+FileName)
        logging.info("Saved clean data to files")
    except Exception as e:
        logging.error("Exception occurred in save_tasksets_data", exc_info=True)
        raise e

    return None


#Data processing Functions
def process_tasksets_Data(df_TaskSet):

	# tranform data values from categorical notation to numeric values

	# df_TaskSet = df_TaskSet.round({'AvgT1': 0, 'AvgT2': 0, 'AvgT3': 0})
	df_TaskSet = df_TaskSet.fillna(0.0)
	# //process arg and PKG value time for task1
	df_TaskSet['Arg1'] = df_TaskSet.Arg1.astype(str)
	df_TaskSet = df_TaskSet.replace({"Arg1": arg_values})
	df_TaskSet = df_TaskSet.replace({"PKG1": pkg_values})

	# //process arg and PKG value time for task2
	df_TaskSet['Arg2'] = df_TaskSet.Arg2.astype(str)
	df_TaskSet = df_TaskSet.replace({"Arg2": arg_values})
	df_TaskSet = df_TaskSet.replace({"PKG2": pkg_values})

	# //process arg and PKG value time for task2
	df_TaskSet['Arg3'] = df_TaskSet.Arg3.astype(str)
	df_TaskSet = df_TaskSet.replace({"Arg3": arg_values})
	df_TaskSet = df_TaskSet.replace({"PKG3": pkg_values})

	return df_TaskSet

def process_db_file(df_tasksets, datasetDirectory_path):
	try:
		logging.info("Load_data")
		logging.info("reading pickle data")

		s = startTimer()
		# reading csv file generated from db
		# df_TaskSet = pd.read_csv(path+saveFile_path+db_cv_file)
		df_TaskSet = df_tasksets

		endTimer("Dataset ", s)

		# drop extra columns
		s = startTimer()
		df_TaskSet = df_TaskSet.drop(columns=All_IDs_taskset)
		df_TaskSet = df_TaskSet.drop(columns=['Period1','Period2','Period3'])
		# df_TaskSet = df_TaskSet.drop(columns=['AVG_RUNTIME1','AVG_RUNTIME2','AVG_RUNTIME3'])

		endTimer("Dropping id's and period columns ", s)

		s = startTimer()
		# column with label/target values is extracted and removed
		df_tasksets_successful_value = df_TaskSet['Successful']
		df_TaskSet = df_TaskSet.drop(columns='Successful')

		endTimer("Create Label Array ", s)

		s = startTimer()
		# Process all data
		df_TaskSet = process_tasksets_Data(df_TaskSet)

		endTimer("Process taskset data ", s)
		# calling save function to save the feature and label set
		save_feature_label_set(datasetDirectory_path, df_TaskSet, df_tasksets_successful_value)
		return df_TaskSet

	except Exception as e:
		logging.info('Error in data loading')
		logging.error("Exception occurred in Load_data", exc_info=True)
		raise e

	return None


def save_feature_label_set(directory_path, df_FeatureSet, df_LabelSet):
	try:
		logging.info("Save data- as csv/pkl format")

		# Saving datas
		s = startTimer()
		# target data
		df_LabelSet.to_csv(path + "Datasets/" + directory_path+"target_data.csv")
		df_LabelSet.to_pickle(path + "Datasets/" + directory_path+"target_data.pkl")

		# feature data
		df_FeatureSet.to_csv(path + "Datasets/" + directory_path+"feature_data.csv")
		df_FeatureSet.to_pickle(path + "Datasets/" + directory_path+"feature_data.pkl")

		# data columns name
		pd.DataFrame({"ColumnName": df_FeatureSet.columns}).to_pickle(path + "Datasets/" + directory_path+"feature_names.pkl")
		pd.DataFrame({"ColumnName": df_FeatureSet.columns}).to_csv(path + "Datasets/" + directory_path+"feature_names.csv")

		target_names_df = pd.DataFrame(Successful, columns=['NOTEXECUTABLE', 'EXECUTABLE'])
		target_names_df.to_pickle(path + "Datasets/" + directory_path+"target_names.pkl")
		target_names_df.to_csv(path + "Datasets/" + directory_path+"target_names.csv")

		endTimer("Saving feature and label data ", s)
		print("------------------------------ save_feature_label_set ------------------------")
	except Exception as e:
		logging.error("Exception occurred in save_feature_label_set", exc_info=True)
		raise e

	return None


if __name__ == "__main__":
	logging.basicConfig(filename=path+'Logging/DataProcessing.log',format='%(asctime)s [%(filename)s: %(funcName)s - %(lineno)d] - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)
	logging.info('==============================================================================')
	logging.info('==============================================================================')
	logging.info('------------------------------main start--------------------------------------')
	logging.info('==============================================================================')
	logging.info('==============================================================================')

	print("Please place DB files inside src/DataProcessing/DB Folder of this project. \n\n")
	print("To extend the code for new db:\n\n")
	print("   - Add new database file inside DB Directory.\n\n")
	print("   - Create new database directory inside Datasets Directory and add the reference of path on top of this file.\n\n")
	print("   - Add the name of database on top of this file  .\n\n")

	print('==============================================================================')
	print('-----------------------  Generate the feature set  --------------------------')
	print('==============================================================================\n')
	print("Select the database:")
	print("type '1' for 'rpi3_final.db' Database")
	print("type '2' for 'panda_v6both.db' Database")
	db_option = input("Enter input: ")
	# set_database_n_quries(db_option)

	if (db_option == '1'):
		print("\t\trpi3_final.db is selected")
		# db name and driectory
		selected_db = dbnamer3
		datasetDirectory_path = datasetDirectory_path_r3
		# db queries
		sql_job_table = sql_job_table_R3
		sql_select_job_table = sql_select_job_table_R3
		sql_task1_table = sql_task1_table_R3
		sql_select_task1_table = sql_select_task3_table_R3
		sql_task2_table = sql_task2_table_R3
		sql_select_task2_table = sql_select_task3_table_R3
		sql_task3_table = sql_task3_table_R3
		sql_select_task3_table = sql_select_task3_table_R3
	else:
		print("\t\tpanda_v6both.db is selected")
		# db name and driectory
		selected_db = dbnamev6
		datasetDirectory_path = datasetDirectory_path_pv6
		# db queries
		sql_job_table = sql_job_table_PV6
		sql_select_job_table = sql_select_job_table_PV6
		sql_task1_table = sql_task1_table_PV6
		sql_select_task1_table = sql_select_task3_table_PV6
		sql_task2_table = sql_task2_table_PV6
		sql_select_task2_table = sql_select_task3_table_PV6
		sql_task3_table = sql_task3_table_PV6
		sql_select_task3_table = sql_select_task3_table_PV6

	# opening database connection
	print('=============== create_db_connection ===========')
	db = create_db_connection("db/"+selected_db)

	print('=============== read_all_data ==================')
	s = startTimer()
	# read all tables from db and concatinate them for generating final taskset
	df_tasksets = read_all_data(db)
	endTimer("read_all_data ", s)
	print(df_tasksets.head(10))

	print('=============== close_db_connection ============')
	# close database connection
	close_db_connection(db)

	print('=============== Process taskset files to generate feature and label set ============')
	# process the dataset array to generate the final feature and label set
	df_feature_data = process_db_file(df_tasksets, datasetDirectory_path)

	print(df_feature_data.head(10))
	print('=============== Ending ==================')
