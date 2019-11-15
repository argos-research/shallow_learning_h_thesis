# shallow_learning_h_thesis


## Technical Details ## 

  Type | Name
------------- | -------------
Language | Python
Machine learning framework | Scikit-learn
IDE | Pycharm
List of packages | Pandas , Sqlite3, Scikit learn, Anaconda, matplotlib

#### Anaconda: #### 
On remote workstations, Anaconda is used for creating the conda environment, which generates a directory that contains a specific collection of conda packages that can be installed by the developers. These packages are necessary for running the python code.
Following command can be run under the anaconda virtual environment: 
* pip install pandas 
* sudo apt-get install sqlite3 
* pip install -U scikit-learn
* pip install logging
* pip install tqdm 
* pip install numpy

# Data Preprocessing: # 

## Databases: ## 

Place the intended database in directory : "/src/DataProcessing/db/"

### Used databases name: ### 
In this project following two databases are used for training:
* panda_v6both.db
* rpi3_final.db


### Databases Access link: ### 
Database for shallow learning based schedulability analysis can be found under following link:
1. For newly generated database:
https://cloud.edvgarbe.de/s/ewaej9gNRy2fTy5

2. For older databases:
https://nextcloud.os.in.tum.de/s/panda_v5
https://nextcloud.os.in.tum.de/s/panda_v4
https://nextcloud.os.in.tum.de/s/panda_v3


## SQL queries ##
In the processing module we have two type of scripts: 
* One which calculates the average, maximum, minimum job runtime values of task per taskset.
  * For rasp3_final database, average, maximum and minimum job runtime is being calculated explicitly.
  
* Second which does not care regarding the average, maximum, minimum job runtime values calculation.
  * For panda_v6 database, average, maximum and minimum job runtime is already given in task table.

This also answers why we have two different SQL queries for PV6 and R3 Dataset.

## Code ##
### Code files ###
All the code related to data processing is present in directory "/src/DataProcessing/" in python file, "Dataset_Processing.py"
### Feature and Label Set Directory ###
Generated Feature set are saved in  following directories
        * "/Datasets/FinalDatasetR3/", used to save raspberry_pi_3 feature set  
        * "/Datasets/FinalDatasetBothV6/", used to save panda_v6both feature set
        
### Runing the Code:###
 In order to use the written script to generatee the feature and label set. Make sure you perform following steps before doing so:
 * Place the db file in 'src/db' folder
 * Make sure that "Dataset_Processing.py" file contains two variable 
    * db file name 
    * feature/label set saving directory path 
 * Make sure root has a directory of Datasets and in it directory for saving the generated feature/label sets
 * After runnng the code, you will be asked to select the intended database 
 * After the running process is complete, feature/label sets are generated automatically in the save directory path, which can be used by the shallow learning classifiers.
  Note: For raspberry_pi_3 and panda_v6both dataset,directories as well as dbname and directory path variables are already created. In order to extend the script more details are present in thesis document under Chapter Implementation, in section 7.6.  

