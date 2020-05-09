#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 6 14:48:43 2020

@author: mehedi
"""


import pandas as pd
import numpy as np
import sys, csv
import time


from sklearn import tree
from sklearn.svm import SVC
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

from RobustPCA.rpca import RobustPCA
from tensorflow import keras

from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score,precision_score, recall_score

from sklearn.model_selection import cross_val_score, cross_validate

import warnings
warnings.filterwarnings("ignore")

from sklearn.decomposition import NMF

import matplotlib.pyplot as plt
from matplotlib import rc
from pylab import *
rcParams.update({'figure.autolayout': True})


# these below lines are very important to match the fonts with actual latex fonts
plt.rc('text', usetex=True)
plt.rc('font', family='Times')

rc('text', usetex=True)

input_path="../data/"
plot_path="../plot/"
output_path= "../output/"

def execute_ml_algorthms(X_train, y_train):
    print("Traditional ML")
    final_scores=[]
    models = ['lr','lda','nb','mlp','svm']
    
    """ Run ML Algorithsm """
    for k in range(len(models)):
        print("Running for:", models[k])
        
        if models[k]=='lr':
            model = LogisticRegression()
            color = 'r-*'
        elif models[k]=='lda':
            model = LinearDiscriminantAnalysis()
            color = 'b-^'
        elif models[k]=='nb':
            model = GaussianNB()
            color = 'g-o'
        elif models[k]=='mlp':
            model = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(3, 2), random_state=1)
            color = 'k--'
        else:
            model = svm.SVC(C=1.0, kernel='rbf', class_weight='balanced')
            color = 'm->'

        cross=10
        scoring = ['accuracy', 'precision_macro', 'recall_macro','f1_macro']
        scores = cross_validate(model, X_train, y_train, scoring=scoring, cv=cross, return_train_score=False)
#        print(scores,"\n\n")

        Training_time= scores['fit_time'].mean()
        Testing_time= scores['score_time'].mean()
        
        """ store the results """
        precision = scores['test_precision_macro']
        recall = scores['test_recall_macro']
        accuracy= scores['test_accuracy']
        fscore= scores['test_f1_macro']

        print("Results: ", accuracy.mean(), precision.mean(), recall.mean(), fscore.mean())
        # sys.exit()
        results=[]
        results.append(models[k])
        results.append(Training_time)
        results.append(Testing_time)
        results.append(accuracy.mean())
        results.append(precision.mean())
        results.append(recall.mean())
        results.append(fscore.mean())
        final_scores.append(results)       
        
    final=pd.DataFrame(final_scores, columns= ['Algorithm', 'Train_Time', 'Test_time', 'Accuracy', 'Precision', 'Recall', 'F1_score'] )
    final.to_csv(output_path+"Traditional_ML_results.csv", index=False)#save in a file

def execute_ACGC(X_train, y_train):
    
    N = len(X_train)
    M = len(X_train[0])
    K = min(M,N)   
    
    """ Non marix factorization used in ACGC """
    model_nmf = NMF(n_components=K, init='random', random_state=0)
    X_train_nmf = model_nmf.fit_transform(X_train)
    
    final_scores=[]
    models = ['lr','lda','nb','mlp','svm']
    
    """ Run ML classifers """
    for k in range(len(models)):
        print("Running for:", models[k])
        
        if models[k]=='lr':
            model = LogisticRegression()
            color = 'r-*'
        elif models[k]=='lda':
            model = LinearDiscriminantAnalysis()
            color = 'b-^'
        elif models[k]=='nb':
            model = GaussianNB()
            color = 'g-o'
        elif models[k]=='mlp':
            model = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(3, 2), random_state=1)
            color = 'k--'
        else:
            model = svm.SVC(C=1.0, kernel='rbf', class_weight='balanced')
            color = 'm->'

        cross=10
        scoring = ['accuracy', 'precision_macro', 'recall_macro','f1_macro']
        scores = cross_validate(model, X_train_nmf, y_train, scoring=scoring, cv=cross, return_train_score=False)
#        print(scores,"\n\n")

        Training_time= scores['fit_time'].mean()
        Testing_time= scores['score_time'].mean()
        
        """ store the results """
        precision = scores['test_precision_macro']
        recall = scores['test_recall_macro']
        accuracy= scores['test_accuracy']
        fscore= scores['test_f1_macro']

        print("Results: ", accuracy.mean(), precision.mean(), recall.mean(), fscore.mean())
        # sys.exit()
        results=[]
        results.append(models[k])
        results.append(Training_time)
        results.append(Testing_time)
        results.append(accuracy.mean())
        results.append(precision.mean())
        results.append(recall.mean())
        results.append(fscore.mean())
        final_scores.append(results)       
        
    final=pd.DataFrame(final_scores, columns= ['Algorithm', 'Train_Time', 'Test_time', 'Accuracy', 'Precision', 'Recall', 'F1_score'] )
    final.to_csv(output_path+"ACGC_results.csv", index=False) #save in a file


def execute_lenet_5(X_train, X_test, X_valid, y_train, y_test, y_valid):
    
    
    channel = 1	
    height=X_train.shape[1]
    #fixing shape 
    X_train = X_train.reshape(-1,height,channel)	
    X_valid = X_valid.reshape(-1,height,channel)	
    X_test =  X_test.reshape(-1,height, channel)	
    
    
    # optimizers=[ "sgd", "Adagrad", "Adam", "Adadelta", "rmsprop"]
    optimizers=["Adam"]
    optimized_accuracy=[]
    optimized_loss=[]
    final_scores=[]
    
    plt.figure(1)
    plt.figsize=(8,5)
    
    for opt in optimizers:
        start = time.clock()
        model = None
        model = keras.models.Sequential([	
	    keras.layers.Conv1D(filters=6, kernel_size=(5), activation="relu",	padding="same",	input_shape=(height,1)),
        
        keras.layers.MaxPooling1D(pool_size=(2)),	
        
		keras.layers.Conv1D(filters=16, kernel_size=(3), activation="relu",	padding="same"),
        		
		keras.layers.AveragePooling1D(),	
        
        keras.layers.Conv1D(filters=120, kernel_size=(3), activation="relu",	padding="same"),	
        
		keras.layers.Flatten(),
        
        keras.layers.Dense(84,activation="relu"),	
		keras.layers.Dense(5,activation="softmax")	])
        
        print(model.summary())	
        
        model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])	
        history=model.fit(X_train, y_train, batch_size=16, epochs=500, validation_data=(X_valid, y_valid))	
        end1 = time.clock()    
        
        # model.evaluate(X_test,	y_test)
        
        
        predictions = model.predict(X_test)
        end2 = time.clock()
        
        Training_time= end1-start
        Testing_time= end2-end1
        
        predictions = predictions.argmax(axis=1)
        
        accuracy=accuracy_score(y_test, predictions)
        precision=precision_score(y_test, predictions,average='macro')
        recall=recall_score(y_test, predictions,average='macro')
        f1=f1_score(y_test, predictions,average='macro')
        
        print(accuracy,precision,recall,f1)
        
        temp_scores=[]
        temp_scores.append(opt)
        temp_scores.append(Training_time)
        temp_scores.append(Testing_time)
        temp_scores.append(accuracy)
        temp_scores.append(precision)
        temp_scores.append(recall)
        temp_scores.append(f1)
        final_scores.append(temp_scores)
        
        optimized_accuracy.append(history.history['acc'])
        optimized_loss.append(history.history['loss'])
        
    for j in range(len(optimized_accuracy)):
        plt.plot(optimized_accuracy[j])
        
    plt.legend(optimizers, loc='best')   
    plt.xlabel("Training Epoch",  fontsize=30)
    plt.ylabel("Training Accuracy",  fontsize=30)
    #plt.show()
    plt.savefig(plot_path+"Training_accuracy.pdf")
    
    
    plt.figure(2)
    for j in range(len(optimized_loss)):
        plt.plot(optimized_loss[j])
        
    plt.legend(optimizers, loc='best')   
    plt.xlabel("Training Epoch", fontsize=30)
    plt.ylabel("Training Loss", fontsize=30)
    plt.savefig(plot_path+"Training_loss.pdf")
    
    with open(output_path+"Lenet_optimizer.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(optimized_accuracy)
        writer.writerows(optimized_loss)
    f.close()
        
    final=pd.DataFrame(final_scores, columns= ['Optimizer', 'Train_time', 'test_time', 'Accuracy', 'Precision', 'Recall', 'F1_score'] )
    
    final.to_csv(output_path+optimizers[0]+"_Lenet5_results.csv", index=False) #save in a file
    
def execute_RPCA(X_train,y_train, iteration=500000):
    
    print("RPCA")
    m=iteration
    
    start=time.clock()
    M= X_train
    
    rpca = RobustPCA(max_iter=m, tol=0.001) #initialize RPCA
    rpca.fit(M) #fit RPCA
    end=time.clock()
    print("\n\nRPCA done\n\n")
    print("Iterations: ",m )
    print("Time: ", (end-start)/60)
    L = rpca.get_low_rank()
    # S = rpca.get_sparse()
    
    final_scores=[]
    models = ['lr','lda','nb','mlp','svm']
    
    """ run ML classifiers """
    for k in range(len(models)):
        print("Running for:", models[k])
        
        if models[k]=='lr':
            model = LogisticRegression()
            color = 'r-*'
        elif models[k]=='lda':
            model = LinearDiscriminantAnalysis()
            color = 'b-^'
        elif models[k]=='nb':
            model = GaussianNB()
            color = 'g-o'
        elif models[k]=='mlp':
            model = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(3, 2), random_state=1)
            color = 'k--'
        else:
            model = svm.SVC(C=1.0, kernel='rbf', class_weight='balanced')
            color = 'm->'

        cross=10
        scoring = ['accuracy', 'precision_macro', 'recall_macro','f1_macro']
        scores = cross_validate(model, L, y_train, scoring=scoring, cv=cross, return_train_score=False) #cross valdiate
#        print(scores,"\n\n")
        Training_time= scores['fit_time'].mean()
        Testing_time= scores['score_time'].mean()
        
        """ store the results """
        precision = scores['test_precision_macro']
        recall = scores['test_recall_macro']
        accuracy= scores['test_accuracy']
        fscore= scores['test_f1_macro']

        print("Results: ", accuracy.mean(), precision.mean(), recall.mean(), fscore.mean())

        # sys.exit()
        results=[]
        results.append(models[k])
        results.append(Training_time)
        results.append(Testing_time)
        results.append(accuracy.mean())
        results.append(precision.mean())
        results.append(recall.mean())
        results.append(fscore.mean())
        final_scores.append(results)       
        
    final=pd.DataFrame(final_scores, columns= ['Algorithm', 'Train_Time', 'Test_time', 'Accuracy', 'Precision', 'Recall', 'F1_score'] )
    final.to_csv(output_path+"RPCA_ML.csv", index=False) #save in a file
    
    
if __name__ == "__main__":
    
    """ Load dataset """
    data = pd.read_csv(input_path+"dacapo.csv")    
    data = data.drop(['Unnamed: 0'], axis=1)
    
    np.random.seed(1798) #fix randomness
     
    data = data.iloc[np.random.permutation(len(data))] #randomized datset
    # reset the index
    data = data.reset_index(drop=True)
    
    class_column = np.array(data.gc_name)
    
    data_column = data.drop(['gc_name'], axis=1)
    data_column = np.array(data_column)
    
    """ divide dataset into train, test, and valdiation set """
    sz = data_column.shape
    
    X_train = data_column[:int(sz[0] * 0.8)]
    X_test  = data_column[int(sz[0] * 0.8):]

    y_train = class_column[:int(sz[0] * 0.8)]
    y_test  = class_column[int(sz[0] * 0.8):]
    
    sz = X_train.shape
    X_valid = X_train[:int(sz[0] * 0.2)]
    y_valid = y_train[:int(sz[0] * 0.2)]
    
    print("Train data shape: ", X_train.shape, y_train.shape)
    print("Validation data shape: ", X_valid.shape, y_valid.shape)
    print("Test data shape: ", X_test.shape, y_test.shape)
    
    print("Options:\n")
    print("1. DL ")
    print("2. RPCA + ML Algorithms")
    print("3. ML Algorithms")
    print("4. ACGC")

    
    user_input = int(input("Enter your option number:  "))
    
    if user_input==1:
        print("DL")
        execute_lenet_5(X_train, X_test, X_valid, y_train, y_test, y_valid)
        print("done")
        
    elif user_input==2:
        print("RPCA")
        iteration = int(input("Enter max iteration number (suggested 500000):  "))
        execute_RPCA(data_column, class_column, iteration)
        print("done")
        
    elif user_input==3:

        print("Traditional ML Algorithms ")
        execute_ml_algorthms(data_column, class_column)
        print("done")
        
    elif user_input==4:
        print("ACGC: ")
        execute_ACGC(data_column, class_column)
        print("done")

        
    else:
        print("Wrong input... please try again!!!!")
        
        

         