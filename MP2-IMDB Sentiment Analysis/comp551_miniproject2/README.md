comp551_miniproject2
==============================

Sentiment analysis for IMDB data for COMP551 McGill University, Winter 2019
Authors: Benjamin MacLellan, John Flores, Mahad Khan

Project Organization
----------------------------------------------------

    ├── README.txt         <- The top-level README for developers using this project.
    ├── final_model.py     <- Final script (PLEASE MARK THIS)
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── test_models.py     <- Dev script (Please DO NOT mark this)
    │
    ├── reports            <- Generated analysis 
    │   ├── figures        <- Generated graphics and figures to be used in reporting
    │   ├── writeup.pdf	   <- Project writeup
    │   └── test_prediction.csv	   <- Kaggle Prediction
    │
    ├── src                <- Source code for use in this project.
    │   │
    │   ├── data           <- Scripts to download or generate data	
    │   │   ├── __pycache__
    │   │   │	├── __init__.cpython-37.pyc
    │   │   │	└── make_dataset.cpython-37.pyc
    │   │   │ 
    │   │   ├── interim
    │   │   │	├── extracted_test_text
    │   │   │	└── extracted_training_text
    │   │   │
    │   │   ├── raw	   <- Location of raw data files (NOT IN ZIP, MUST BE INSERTED)
    │   │   │	├── test   <- Testing data used to generate Kaggle csvs
    │   │   │	└── train  <- Training data
    │   │   │
    │   │   ├── make_dataset.py
    │   │   └── __init__.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   ├── build_features.py
    │   │   └── __init__.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── custom_models.py
    │   │   └── test_prediction.csv
    │   │
    │   └── visualization  <- Scripts to results-oriented visualizations
    │       ├── visualize.py 
    │ 	    └── clean_data.py
    └──

Packages Used:
numpy
scipy
pandas
matplotlib
sklearn
time
os
re
nltk
------------------------------------------------



Important Notes
------------------------------------------------
- The raw data files are NOT in this zip file. They should be put in /src/data/raw, as detailed in the above project organization

Bernoulli Naive Bayes from Scratch (BNBS) 
------------------------------------------------
To read code:
-> Open code location: .\comp551_miniproject2\src\models)

To run BNBS:
-> Run final model.py. Results for Naive Bayes will display on the console. 
-> Script will stop until prompted


Replication of Other Classifier Models 
------------------------------------------------
Available Pipelines:
pipe_tfidf_lr: CountVectorizer -> TfidfTransformer -> Normalizer -> LogisticRegression
pipe_tfidf_dt: CountVectorizer -> TfidfTransformer -> Normalizer -> DecisionTreeClassifier

Specific Sections:
- Section "Run Custom Pipeline" runs our best performing model
- Section "Runtime Tests 2" runs the 6 models tested in Figure 1 of our report
- Section "C-Value Tests 3" runs the C-value, max_iter = 100 models tested in Figure 2 of our report
- Section "Max-Iter Tests 4" runs the max_iter models tested in Figure 3 of our report.
- Section "C-Value Tests 5" runs the C-value, max_iter = 150 models tested in Figure 2 of our report

--------
To replicate results from report (text files, not graphs): 
0) Change working directory to comp551_miniproject2
1) Run final_model.py
2) Go to .\comp551_miniproject2\src\models
3) 5 folders will have been created. Open the run folder that was created first. For each pipeline in pipeline_grid, there will be 3 text files: cv_results_{}, test_report_{}, and validation_test_report_{} 
	-> cv_results is a raw report of the results of the grid search. Hard to interpret, but can be loaded into python to extract more information.
	-> test_report is a cleaned pipeline-by-pipeline list of results including mean validation score, stdev, and time to run the full pipeline. Information from test_report was used in the written report.
	-> validation_test_report is the f1 matrix of the top-performing hyperparameters in the tested pipeline
4) rt_results.txt contains a list of runtimes for the full pipeline
5) Similar files are generated in the other folders

--------
To set up custom pipeline:
1) Scroll down to Section "Run Custom Pipeline"
2) Search for variable "pipeline_grid"

    -> pipeline_grid is a dictionary with;
	- key = Identifying string for trial
	- entry = [pipeline, pipeline parameters]
    -> Available pipelines are listed above
    -> Pipeline parameters are a dictionary of parameters to set in pipeline
	- See commented variable: rt_params_dt as an example. 

3) Define a parameter variable with desired list of parameters
4) Choose pipeline
5) Add dictionary entry to pipeline_grid
6) Write in: run_pipeline(pipeline_grid,5,n), where n is the number of parameter permutations
7) Run "Run Custom Pipeline (Best Model)" Section

--------
To generate C Value and Max_Iter Graphs:
1) Run final_model.py
2) Navigate to src/visualization/clean_data.py
3) Set filename_from to correct C-value
4) Uncomment the appropriate filename_to
5) Run the script
6) Run visualize.py, in the same folder

--------
To generate Model Comparisons Graph (Note, this is manual):
1) Navigate to model_test_results.csv
2) Manually add entries according to test_report_{}.txt in the relevant folder in /model
3) Run visualize.py, in the same folder

*Note to TA: It may be easier to just look at the text files generated in /src/models/run-xxxx to read off the MVS values

--------
To generate Kaggle test set
While Method 1 is significantly faster, method 2 can be done during a run of the entire code. 
test_prediction.csv will be generated in reports folder

Method 1:
1) Run the "Import Packages" and "Import Data" sections of the code
2) Scroll down to the bottom and run "Run on Test Files" section of code

Method 2:
1) Run final_model.py fully. 
2) Near the end of the run, follow the prompts to create test_models.csv


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
