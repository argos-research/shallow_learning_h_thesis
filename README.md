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
        
### Runing the Code ###
In order to use the written script to generatee the feature and label set. Make sure you perform following steps before doing so:
* Place the db file in 'src/db' folder
* Make sure that "Dataset_Processing.py" file contains two variable 
  * db file name 
  * feature/label set saving directory path 
* Make sure root has a directory of Datasets and in it directory for saving the generated feature/label sets
* After runnng the code, you will be asked to select the intended database 
* After the running process is completed, feature/label sets are generated automatically in the save directory path, which can be used by the shallow learning classifiers.

Note: For raspberry_pi_3 and panda_v6both dataset,directories as well as dbname and directory path variables are already created. In order to extend the script more details are present in thesis document under Chapter Implementation, in section 7.6.  

  
  
# Shallow learning classifier: # 

## Directory ## 
* Logging
    * Custom Logging- "root/BestModels"
      Only the best models of hyper parameter optimization are stored in the custom file.
    * Default Logging - "root/Logging"
      This uses default system logger

* Model - "root/src/Models"
  This contains a seperate class for each type of classifier. In these classes all the function related to training, evaluation and saving the model are present.
    * Decision tree
    * Random forest
    * K nearest neighbors
    * Logistic regression
    * Support vector machine
  
* ‘ShallowLearning_Training.py’:
  Contain the main file for running the training process.

* ‘Log_BestModel.py':
  This class contains custom logging functions for each type of classifier. 
   
  ## Code ## 
  ### Main Program ###
  The main program function call is implemented in the file called ‘ShallowLearning_Training.py’. 
  
  ### Parameter Setting ###
  * hyper parameter optimization
  For hyper parameter optimization, parameter array can be defined inside each class:
    * Inside class add the values in the existing array called "parameter_sets"
  
  * Single model training 
     For training the single model,parameters can be set inside the constructor of each class. 
  Note: Hyper parameter used in this project are already added inside the each class of shallow learning algorithm. Those parameter can be taken as reference point for future work. 
  ## Running the code ##
  * Make sure you have added the following thing in the source  code
    * Name and path of dataset are adding on top of ‘ShallowLearning_Training.py’ file.
    * Make sure Dataset Directory and feature/label set exist,
    * Make sure Logging and custom logging files are presets in the source code for each type of classsifier. 

  * Main code
    * ‘ShallowLearning_Training.py’. is run for the execution of training, on command line user will be asked to 
      * select the dataset
      * select the shallow learning classifier - SVM, KNN etc. 
      * select the type of training- Hyperparameter optimization or single Model
      After which training will automatically be started. Depending on the parameter and CPU cores, training time will be varying and results will be produced accordingly. 
  
  * Then training is automatically started  once training is finished results can be viewed on console as well as inside the logging files. For hyper paprameter custom and default both logging file should be viewd for complete overview of training process and results generated in the end. For single model training, everything is displayed on the console. 

 Note: In order to extend the script for adding new classifier further details can be viewed inside the thesis document under Chapter Implementation, in section 7.6.  
