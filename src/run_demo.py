#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 23:48:53 2020

@author: mq20197379

This demo is to show how to use the LSHiForest model for anomaly detection. LSHiForest can use several types of distance metrics.
The IsolationForest model from scikit-learn is used for comparison. Note that this model is a special case of LSHiForest with a standardized L1 distance.

The features in the 'glass.csv' data set has been preprocessed with standardization. The anomaly data instances are with label '-1', while the normal data instances are with label '1'.

"""

import sys
print(sys.version)

import pandas as pd
import time

from sklearn.metrics import roc_auc_score
from sklearn.ensemble import IsolationForest

from detectors import LSHiForest

import warnings
warnings.filterwarnings("ignore")

""""""
num_ensemblers=100

glass_df = pd.read_csv('../dat/glass.csv', header=None)
X = glass_df.values[:, :-1]
ground_truth = glass_df.values[:, -1]

#classifiers = [("L2SH", LSHiForest())]

classifiers = [("sklearn.ISO", IsolationForest(num_ensemblers)), ("ALSH", LSHiForest('ALSH', num_ensemblers)), ("L1SH", LSHiForest('L1SH', num_ensemblers)), ("L2SH", LSHiForest('L2SH', num_ensemblers)), ("KLSH", LSHiForest('KLSH', num_ensemblers))]


for i, (clf_name, clf) in enumerate(classifiers):
	
	print("\n"+clf_name+":")
	start_time = time.time()
	
	clf.fit(X)
	
	train_time = time.time()-start_time
	
	y_pred = clf.decision_function(X)
	
	test_time = time.time()-start_time-train_time
	
	auc = roc_auc_score(ground_truth, y_pred)
	
	print("\tAUC score:\t", auc)
	print("\tTraining time:\t", train_time) 
	print("\tTesting time:\t", test_time)