#! /usr/bin/python
#
# Implemented by Xuyun Zhang (email: xuyun.zhang@auckland.ac.nz). Copyright reserved.
#
import sys
print(sys.version)

import numpy as np
import pandas as pd
import csv
import time
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import IsolationForest

from detectors import VSSampling
from detectors import Bagging
from detectors import LSHForest
from detectors import E2LSH, KernelLSH, AngleLSH


rng = np.random.RandomState(42)
num_ensemblers=100

data = pd.read_csv('../dat/glass.csv', header=None) 
X = data.as_matrix()[:, :-1].tolist()
ground_truth = data.as_matrix()[:, -1].tolist()
	
classifiers = [("sklearn.ISO", IsolationForest(random_state=rng)), ("ALSH", LSHForest(num_ensemblers, VSSampling(num_ensemblers), AngleLSH())), ("L1SH", LSHForest(num_ensemblers, VSSampling(num_ensemblers), E2LSH(norm=1))), ("L2SH", LSHForest(num_ensemblers, VSSampling(num_ensemblers), E2LSH(norm=2))), ("KLSH", LSHForest(num_ensemblers, VSSampling(num_ensemblers), KernelLSH()))]

for i, (clf_name, clf) in enumerate(classifiers):
	print("	"+clf_name+":")
	start_time = time.time()
	clf.fit(X)
	train_time = time.time()-start_time
	y_pred = clf.decision_function(X).ravel()
	test_time = time.time()-start_time-train_time
	auc = roc_auc_score(ground_truth, -1.0*y_pred)
	print("AUC:	", auc)
	print("Training time:	", train_time) 
	print("Testing time:	", test_time)
