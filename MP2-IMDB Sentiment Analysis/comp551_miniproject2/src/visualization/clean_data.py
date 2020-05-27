# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 01:56:43 2019

@author: jpflo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


filename_from = '../models/run_0222-1623/test_report_Pipe_30_DT_Runtime.txt'
#filename_to = 'C-Value-100.csv'
#filename_to = 'C-Value-150.csv'
filename_to = 'N_ITER-Value.csv'

def clean_value_report(f1,f2):
    
    with open(f1, newline='') as csvfile:
        df1 = pd.read_table(csvfile)
        
    df2 = list(df1.values)
    
    df3 = [d[0].split(",") for d in df2]
    
    df4 = [[float(d[1].replace(" {'clf__C': ",'')),
            float(d[2].replace(" 'clf__max_iter': ",'')),
            np.log10(float(d[1].replace(" {'clf__C': ",''))),
            float(d[-3].replace(" ",'')),
            float(d[-2].replace(" ",'')),] 
        for d in df3]
    
    f = open(f2,'w')
    f.write('C, n_iter,log(C),Mean,Error \n')
    f.close()
    
    f = open(f2,'a')
    
    for d in df4:
        f.write(str(d).replace("[",'').replace(']','') + '\n')
        
    f.close()
    
clean_value_report(filename_from,filename_to)
    