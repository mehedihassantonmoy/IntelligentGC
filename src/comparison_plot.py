#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 22:16:01 2020

@author: mehedi
"""

import pandas as pd
import numpy as np
import sys, csv
import time

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import matplotlib.font_manager
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['libertine']})
rc('font',**{'family':'serif','serif':['Libertine']})
rc('text', usetex=True)

plt.rcParams["figure.figsize"] = (13.1,8)


input_path="../data/"
plot_path="../plot/"
output_path= "../output/"

if __name__ == "__main__":
    """ Load results """
    ACGC_results = pd.read_csv(output_path+"ACGC_results.csv")  
    Lenet5_results = pd.read_csv(output_path+"Adam_Lenet5_results.csv")  
    RPCA_ML = pd.read_csv(output_path+"RPCA_ML.csv")  
    Traditional_ML_results = pd.read_csv(output_path+"Traditional_ML_results.csv")  
    
    # Traditional_ML_results.iloc[:, 3:].idxmax(axis=0)
    # ACGC_results.iloc[:, 3:].idxmax(axis=0)
    
    """ Extract accuracy, presion, recall and f1 """
    ACGC_results_time= ACGC_results.iloc[4, 3:]
    Traditional_ML_results_time= Traditional_ML_results.iloc[1, 3:]
    RPCA_ML_time= RPCA_ML.iloc[1, 3:]
    Lenet5_results_time= Lenet5_results.iloc[0, 3:]
    
    """ bar plot for comparison"""
    N = 4
    
    ind = np.arange(N)
    width = 0.1
    fig, ax = plt.subplots()
    
    """ fit data in rects """
    rects1 = ax.bar(ind, Traditional_ML_results_time.values, width)
    
    rects2 = ax.bar(ind+width,  ACGC_results_time.values, width)

    rects3 = ax.bar(ind + 2 * width, RPCA_ML_time.values, width)

    rects4 = ax.bar(ind + 3 * width, Lenet5_results_time.values, width)
    
    ax.set_ylabel('Scores', fontsize=30)

    ax.set_xticks(ind + 1.41*width)
    ax.set_xticklabels(('Accuracy', 'Precision', 'Recall', 'F1'), fontsize=20)
    ax.legend((rects1[0], rects2[0], rects3[0], rects4[0]), ('LDA', 'ACGC + SVM', 'IntelligentGC+ RPCA +LDA', 'IntelligentGC+ LeNet5'), loc=8)
    # ax.set_ylim(0, 0.5)
    
    plt.show()
    plt.savefig("comparison.eps")