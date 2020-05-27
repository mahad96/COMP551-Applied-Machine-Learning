#%% Import Packages
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.decomposition import PCA
import pickle

from src.models.custom_models import CustomBernoulliNaiveBayes
from src.data.make_dataset import raw_data_extraction

from sklearn.feature_extraction.text import CountVectorizer

import time

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline


from sklearn import metrics 
from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV
from os import mkdir

#%% Import Data

import_bool = input("Do you want to load from ./src/data/interim/extracted_training_text.txt? (1 for yes, 0 for no)")

if (not import_bool):

    print('Importing Data')
    directories = ['./src/data/raw/train/neg/', './src/data/raw/train/pos/']
    raw_text_lst, raw_target_lst, raw_text_id = raw_data_extraction(directories)

    print('Setting to Pandas Dataframe')
    raw_data = pd.DataFrame({'target':raw_target_lst, 'raw_text':raw_text_lst})
    raw_data.to_pickle('./src/data/interim/extracted_training_text')

else:
    raw_data = pd.read_pickle('./src/data/interim/extracted_training_text')
    
## reduce how many examples we use, if wanted (should run fine with full data set)
data = raw_data.sample(frac=1.0)

#%% Naive Bayes

print('Extracting Raw Data')

## count words and binarize (1 if word appears, 0 if not)
#corpus = data['raw_text'].to_numpy()
corpus = np.array(data['raw_text'])
vectorizer = CountVectorizer()
word_counts_raw = vectorizer.fit_transform(corpus)
word_counts = word_counts_raw#.todense()


## some extra features for future
#vectorizer = TfidfVectorizer()
#data = count_past_tense_verbs(data)

print('Setting Up Training and Validation Data for Naive Bayes')
## set up train/valid data
y = np.array(data['target'])
X = word_counts.astype(bool)

X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, test_size=0.2, random_state=0)

print('Initializing Naive Bayes Model')

## initialize the model
clf = CustomBernoulliNaiveBayes(laplaceSmoothing = True)  # our written model, can use all the SKLearn functions with it (see src/models/custom_models for the code)
#clf = DecisionTreeClassifier(random_state=0)
#clf = SVC(gamma=0.001, C=100.)
#clf = BernoulliNB()  # to compare our custom model with - we get identical results, but slightly slower

## train once and predict
t1 = time.time()
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print('Score: {}'.format(score))

## do k-fold cross-validation

cv = cross_val_score(clf, X, y, cv = 5)
print('Mean for cross-validation: {}, Individual: {}'.format(np.mean(cv), cv))

t2 = time.time()
print('Time to train and predict/cross-validate: {}'.format(t2-t1))

input("Input any key to continue.")
#%% Pipelines

# Set Up Data
print('Splitting Data into Training and Validation Sets for Pipelines')
X_train, X_test, y_train, y_test = train_test_split(data['raw_text'], 
                data['target'], train_size=0.8, test_size=0.2, random_state=0)

# Utility Function to Report Best Scores
# From https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html
def report(results, n_top, trial, runtime, path):
    
    filename = "./src/models/" + path + "/test_report_" + trial + ".txt"
    f = open(filename,"a")
    f.write("Trial, Parameters, Max Validation Score, St Dev, Time\n")
    
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        top_candidate = np.flatnonzero(results['rank_test_score'] == 1)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
            
            f.write(trial + ", " + 
                    str(results['params'][candidate]) + ", " +
                    str(results['mean_test_score'][candidate]) + ", " +
                    str(results['std_test_score'][candidate]) + ", " +
                    str(runtime) + "\n")
            
    f.close()
    
    return results['params'][top_candidate[0]]



## Feature Extraction Pipelines
# Logistic Regression
pipe_tfidf_lr = Pipeline([('vect', CountVectorizer() ),
                          ('tfidf', TfidfTransformer() ),
                          ('norm', Normalizer() ),
                          ('clf', LogisticRegression() )])


# Decision Trees
pipe_tfidf_dt = Pipeline([('vect', CountVectorizer() ),
                          ('tfidf', TfidfTransformer() ),
                          ('norm', Normalizer() ),
                          ('clf', DecisionTreeClassifier() )])

# Outdated --------------------------------------------------------------------
    
#pipeline_list = {#"Pipe_1_BO-LR": pipe_bo_lr
#                 "Pipe_2_TFIDF-LR": pipe_tfidf_lr, 
#                 "Pipe_3_BO-DT": pipe_bo_dt, 
#                 "Pipe_4_TFIDF-DT": pipe_tfidf_dt
#                 "Pipe_5_BO-NGRAM-LR": pipe_bo_lr
#                 "Pipe_6_TRUE_BO-NGRAM-LR": pipe_tfidf_lr,
#                  "Pipe_7_BO-TFIDF-LR": pipe_tfidf_lr}
#                  "Pipe_8_BO-TFIDF-LR": pipe_tfidf_lr}
#                  "Pipe_11_BO-TFIDF-LR": pipe_tfidf_lr}

#tfidf_params = {"vect__ngram_range": [(1,1)], 
#                "tfidf__use_idf": [True]}
#
#dt_params = {"clf__min_impurity_decrease": [1e-5], 
#             "clf__min_samples_split": [2,3,4]}
#
#lr_params = {"clf__tol":[1e-4], "clf__solver":["lbfgs"]}

print('Defining Parameters')


bo_params = {"vect__ngram_range": [(1,1),(1,2),(1,3),(2,2),(2,3),(3,3)],
                                   "vect__binary": [True,False]}

tfidf_params = {"vect__ngram_range": [(1,1),(1,2),(1,3),(2,2),(2,3),(3,3)], 
                "tfidf__use_idf": [True,False]}

dt_params = {"clf__min_impurity_decrease": [1e-5,5e-6,2e-6,1e-6], 
             "clf__min_samples_split": [2,5,10]}

lr_params = {"clf__tol":[1e-4,1e-5,5e-6,1e-6], "clf__solver":["lbfgs"]}


bo_lr_params = lr_params.copy()
tfidf_lr_params = lr_params.copy()

bo_dt_params = dt_params.copy()
tfidf_dt_params = dt_params.copy()

bo_lr_params.update(bo_params)
bo_dt_params.update(bo_params)

tfidf_lr_params.update(tfidf_params)
tfidf_dt_params.update(tfidf_params)

# Outdated Test Parameters ----------------------------------------------------
#bo_tfidf_lr_params = {"vect__binary": [True], 
#                      "vect__ngram_range": [(1,2)],
#                      "tfidf__use_idf": [True],
#                      "clf__solver": ['lbfgs'],
#                      "clf__C": [1,10,100,1000]}
#
#bo_tfidf_lr_params_2 = {"vect__binary": [True], 
#                      "vect__ngram_range": [(1,2)],
#                      "tfidf__use_idf": [True],
#                      "clf__solver": ['lbfgs'],
#                      "clf__C": [2000,5000,10000,20000,50000]}
#
#bo_tfidf_lr_params_3 = {"vect__binary": [True], 
#                      "vect__ngram_range": [(1,2)],
#                      "tfidf__use_idf": [True],
#                      "clf__solver": ['lbfgs'],
#                      "clf__C": [10000],
#                      "clf__max_iter": [100,150,200,250,300] }
#
#
#bo_tfidf_lr_params_4 = {"vect__binary": [True], 
#                      "vect__ngram_range": [(1,2)],
#                      "tfidf__use_idf": [True],
#                      "clf__solver": ['lbfgs'],
#                      "clf__C": [8600,8700,8800,8900,9100,9200,9300,9400],
#                      "clf__max_iter": [150] }
#
#bo_tfidf_lr_params_5 = {"vect__binary": [True], 
#                      "vect__ngram_range": [(1,2),(1,3)],
#                      "tfidf__use_idf": [True],
#                      "clf__solver": ['lbfgs'],
#                      "clf__C": [10000],
#                      "clf__max_iter": [150] }
#
#bo_tfidf_lr_params_6 = {"vect__binary": [True], 
#                      "vect__ngram_range": [(1,2),(2,2)],
#                      "tfidf__use_idf": [True],
#                      "clf__solver": ['lbfgs'],
#                      "clf__C": [50, 75, 80, 90, 100, 150, 200, 500, 750, 
#                                 900, 1000, 2000, 5000, 7500, 9000, 10000],
#                      "clf__max_iter": [150] }
#
#bo_tfidf_lr_params_7 = {"vect__binary": [True], 
#                      "vect__ngram_range": [(1,2)],
#                      "tfidf__use_idf": [True],
#                      "clf__solver": ['lbfgs'],
#                      "clf__C": [8000,8500],
#                      "clf__max_iter": [150] }
#
#param_master_grid = {#"Pipe_1_BO-LR": lr_params,
#                     #"Pipe_2_TFIDF-LR": tfidf_lr_params,
#                     #"Pipe_3_BO-DT": bo_dt_params,
#                     #"Pipe_4_TFIDF-DT": tfidf_dt_params
#                     #"Pipe_6_TRUE_BO-NGRAM-LR": bo_lr_params
#                     "Pipe_11_BO-TFIDF-LR": bo_tfidf_lr_params_4}
# -----------------------------------------------------------------------------

#%% Pipeline Function


def run_pipeline(pipeline_grid,cvnum,num_of_trials_per_pipe):
    
    path = "run_" + time.strftime("%m%d-%H%M")
    mkdir("./src/models/{}".format(path))
    
    filename4 = "./src/models/{}/rt_results".format(path) + ".csv"
    f4 = open(filename4,'a')
    f4.write("Pipe, Runtime /n")
    f4.close()
    
    for pipe in pipeline_grid:
        print('Performing Grid Search for {}'.format(pipe))
        grid_search = GridSearchCV(pipeline_grid[pipe][0], 
                                   param_grid = pipeline_grid[pipe][1], 
                                   cv=cvnum)
        
        start = time.time()
        grid_search.fit(X_train, y_train)
        end = time.time()
        
        y_pred = grid_search.predict(X_test)
        
        runtime = end - start
        print("GridSearchCV for {} took {} seconds".format(pipe,runtime))
        
        # Change the integer according to the number of pipeline results to print
        top_params = report(grid_search.cv_results_, num_of_trials_per_pipe, pipe, runtime, path)
        
        cl_rpt = metrics.classification_report(y_test,y_pred, target_names=('Negative','Positive'))
        print(cl_rpt)
        
        filename = "./src/models/{}/validation_test_report_{}".format(path,pipe) + ".txt"
        f = open(filename,'a')
        f.write(str(top_params) + "\n")
        f.write(cl_rpt)
        f.close()
        
        filename2 = "./src/models/{}/cv_results_{}".format(path,pipe) + ".txt"
        f2 = open(filename2,'a')
        f2.write(str(grid_search.cv_results_))
        f2.close()
        
        filename4 = "./src/models/{}/rt_results".format(path) + ".csv"
        f4 = open(filename4,'a')
        f4.write("{}, {} \n".format(pipe,runtime))
        f4.close()
        
#%% Run Custom Pipeline (Best Model)
        
rt_params_lr = {"vect__binary": [True], 
               "vect__ngram_range": [(1,2)],
               "tfidf__use_idf": [True],
               "clf__solver": ['lbfgs'],
               "clf__C": [7500],
               "clf__max_iter": [150] }

#rt_params_dt = {"vect__binary": [True], 
#               "vect__ngram_range": [(1,2)],
#               "tfidf__use_idf": [True]}

# Entry: "Key": [Pipeline, Parameter Grid]

# Example Grid
# pipeline_grid = {"Pipe_1_BO-LR": [pipe_bo_lr, lr_params],
#                  "Pipe_2_TFIDF-LR":[pipe_tfidf_lr, tfidf_lr_params],
#                  "Pipe_3_BO-DT": [pipe_bo_dt, bo_dt_params],
#                  "Pipe_4_TFIDF-DT": [pipe_tfidf_dt, tfidf_dt_params],
#                  "Pipe_6_TRUE_BO-NGRAM-LR": [pipe_bo_lr, bo_lr_params],
#                  "Pipe_7_BO-TFIDF-LR": [pipe_bo_lr, bo_tfidf_lr_params_4]}

pipeline_grid = {"Pipe_15_RT_Comp_LR": [pipe_tfidf_lr, rt_params_lr]}

run_pipeline(pipeline_grid,5,1)
    

#%% Runtime Tests 1
    
#rt_params_dt_1 = {"vect__binary": [False], 
#                  "vect__ngram_range": [(1,1)],
#                  "tfidf__use_idf": [False]}
#
#rt_params_lr_2 = {"vect__binary": [False],
#                  "vect__ngram_range": [(1,1)],
#                  "tfidf__use_idf": [False],
#                  "clf__solver": ['lbfgs'] }
#
#rt_params_lr_3 = {"vect__binary": [False],
#                  "vect__ngram_range": [(1,2)],
#                  "tfidf__use_idf": [False],
#                  "clf__solver": ['lbfgs']}
#
#rt_params_lr_4 = {"vect__binary": [True], 
#                  "vect__ngram_range": [(1,2)],
#                  "tfidf__use_idf": [False],
#                  "clf__solver": ['lbfgs'] }
#
#rt_params_lr_5 = {"vect__binary": [False],
#                  "vect__ngram_range": [(1,1)],
#                  "tfidf__use_idf": [True],
#                  "clf__solver": ['lbfgs']}
#
#rt_params_lr_6 = {"vect__binary": [True], 
#                  "vect__ngram_range": [(1,2)],
#                  "tfidf__use_idf": [True],
#                  "clf__solver": ['lbfgs']}
#    
#pipeline_grid = {"Pipe_17_DT_Runtime": [pipe_tfidf_dt, rt_params_dt_1],
#                 "Pipe_18_LR_RUntime": [pipe_tfidf_lr, rt_params_lr_2],
#                 "Pipe_19_LR_RUntime": [pipe_tfidf_lr, rt_params_lr_3],
#                 "Pipe_20_LR_RUntime": [pipe_tfidf_lr, rt_params_lr_4],
#                 "Pipe_21_LR_RUntime": [pipe_tfidf_lr, rt_params_lr_5],
#                 "Pipe_22_LR_RUntime": [pipe_tfidf_lr, rt_params_lr_6]}
#
#run_pipeline(pipeline_grid,5,1)

#%% Runtime Tests 2
    
rt_params_dt_1 = {"vect__binary": [False], 
                  "vect__ngram_range": [(1,1)],
                  "tfidf__use_idf": [False]}

rt_params_lr_2 = {"vect__binary": [False],
                  "vect__ngram_range": [(1,1)],
                  "tfidf__use_idf": [False],
                  "clf__solver": ['lbfgs'] }

rt_params_lr_3 = {"vect__binary": [False],
                  "vect__ngram_range": [(1,2)],
                  "tfidf__use_idf": [False],
                  "clf__solver": ['lbfgs']}

rt_params_lr_4 = {"vect__binary": [True], 
                  "vect__ngram_range": [(1,1)],
                  "tfidf__use_idf": [False],
                  "clf__solver": ['lbfgs'] }

rt_params_lr_5 = {"vect__binary": [False],
                  "vect__ngram_range": [(1,1)],
                  "tfidf__use_idf": [True],
                  "clf__solver": ['lbfgs']}

rt_params_lr_6 = {"vect__binary": [True], 
                  "vect__ngram_range": [(1,2)],
                  "tfidf__use_idf": [True],
                  "clf__solver": ['lbfgs']}
    
pipeline_grid = {"Pipe_23_DT_Runtime": [pipe_tfidf_dt, rt_params_dt_1],
                 "Pipe_24_LR_Runtime": [pipe_tfidf_lr, rt_params_lr_2],
                 "Pipe_25_LR_Runtime": [pipe_tfidf_lr, rt_params_lr_3],
                 "Pipe_26_LR_Runtime": [pipe_tfidf_lr, rt_params_lr_4],
                 "Pipe_27_LR_Runtime": [pipe_tfidf_lr, rt_params_lr_5],
                 "Pipe_28_LR_Runtime": [pipe_tfidf_lr, rt_params_lr_6]}

run_pipeline(pipeline_grid,5,1)

#%% C-Value Tests 3

rt_params_lr_7 = {"vect__binary": [True], 
                  "vect__ngram_range": [(1,2)],
                  "tfidf__use_idf": [True],
                  "clf__solver": ['lbfgs'],
                  "clf__max_iter":[100],
                  "clf__C":[1e-3,1e-2,1e-1,1,5,10,50,100,500,1000,5000,10000]}
    
pipeline_grid = {"Pipe_32_LR_C-Value": [pipe_tfidf_lr, rt_params_lr_7]}

run_pipeline(pipeline_grid,5,12)

#%% Max-Iter Tests 4

rt_params_lr_8 = {"vect__binary": [True], 
                  "vect__ngram_range": [(1,2)],
                  "tfidf__use_idf": [True],
                  "clf__solver": ['lbfgs'],
                  "clf__C":[7500],
                  "clf__max_iter":[100,150,200,250,300]}
    
pipeline_grid = {"Pipe_30_DT_Runtime": [pipe_tfidf_lr, rt_params_lr_8]}

run_pipeline(pipeline_grid,5,5)

#%% C-Value Tests 5

rt_params_lr_9 = {"vect__binary": [True], 
                  "vect__ngram_range": [(1,2)],
                  "tfidf__use_idf": [True],
                  "clf__solver": ['lbfgs'],
                  "clf__C":[1e-3,1e-2,1e-1,1,5,10,50,100,500,1000,5000,10000],
                  "clf__max_iter":[150]}
    
pipeline_grid = {"Pipe_29_LR_C-Value": [pipe_tfidf_lr, rt_params_lr_9]}

run_pipeline(pipeline_grid,5,12)

#%% Run on Test Files
from src.data.make_dataset import raw_data_extraction_test_set

gen_pred = input('Do you want to generate a Kaggle CSV File? (1 for yes, 0 for no)')

if gen_pred:
    imp_test = input('Do you want to load from ./src/data/interim/extracted_test_text? (1 for yes, 0 for no)')
    
if (not imp_test):
  
    test_directories = ['./src/data/raw/test/']
    raw_text_lst, raw_text_id = raw_data_extraction_test_set(test_directories)
    
    raw_test_data = pd.DataFrame({'id':raw_text_id, 'raw_text':raw_text_lst})
    raw_test_data.to_pickle('./src/data/interim/extracted_test_text')

else:
    raw_test_data = pd.read_pickle('./src/data/interim/extracted_test_text')

if gen_pred:
    test_pipeline =  Pipeline([('vect', CountVectorizer()),
                              ('tfidf', TfidfTransformer() ),
                              ('norm', Normalizer() ),
                              ('clf', LogisticRegression() )]) 
    
    test_params = {"vect__binary": [True], 
                   "vect__ngram_range": [(1,2)],
                   "tfidf__use_idf": [True],
                   "clf__solver": ['lbfgs'],
                   "clf__C": [7500],
                   "clf__max_iter": [150] }
    
    grid_search = GridSearchCV(test_pipeline, 
                               param_grid = test_params, 
                               cv=5)
    
    # Fit to entire training data set
    grid_search.fit(data['raw_text'],data['target'])
    
    
    y_test_pred = grid_search.predict(raw_test_data['raw_text'])
    
    filename3 = "./reports/test_prediction.csv"
    f3 = open(filename3,"a")
    
    
    f3.write("Id,Category\n")
    
    for i in range(len(y_test_pred)):
        f3.write(raw_test_data['id'][i] + "," + str(y_test_pred[i]) + "\n")
        
    f3.close()