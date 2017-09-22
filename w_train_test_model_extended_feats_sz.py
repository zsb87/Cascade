#
# DISCLAIMER
#
# This script is copyright protected 2015 by
# Edison Thomaz, Irfan Essa, Gregory D. Abowd
#
# All software is provided free of charge and "as is", without
# warranty of any kind, express or implied. Under no circumstances
# and under no legal theory, whether in tort, contract, or otherwise,
# shall Edison Thomaz, Irfan Essa or Gregory D. Abowd  be liable to
# you or to any other person for any indirect, special, incidental,
# or consequential damages of any character including, without
# limitation, damages for loss of goodwill, work stoppage, computer
# failure or malfunction, or for any and all other damages or losses.
#
# If you do not agree with these terms, then you are advised to 
# not use this software.
#


from __future__ import division
import time
import datetime
import csv
from sklearn import svm, neighbors, metrics, cross_validation, preprocessing
from sklearn.externals import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, silhouette_score
from sklearn.cluster import KMeans, DBSCAN
from scipy import *
from scipy.stats import *
from scipy.signal import *
from numpy import *


import os
import re
import matplotlib
import numpy as np
import scipy.fftpack
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
from collections import Counter
from sklearn import preprocessing
from sklearn.metrics import matthews_corrcoef
import numpy.polynomial.polynomial as poly
# import plotly 
pd.set_option('display.max_rows', 500)

from sklearn import preprocessing
from sklearn.metrics import auc, silhouette_score
from collections import Counter
from sklearn.metrics import *
from sklearn.metrics import precision_recall_fscore_support as score



def calc_cm(y_test, y_pred):#

    
    # ct = pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True).apply(lambda r: r/r.sum(), axis=1)
    ct = pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
    print(ct)
    # ct.to_csv(cm_file)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    accuracy = sum(cm[i,i] for i in range(len(set(y_test))))/sum(sum(cm[i] for i in range(len(set(y_test)))))
    recall_all = sum(cm[i,i]/sum(cm[i,j] for j in range(len(set(y_test)))) for i in range(len(set(y_test))))/(len(set(y_test)))
    precision_all = sum(cm[i,i]/sum(cm[j,i] for j in range(len(set(y_test)))) for i in range(len(set(y_test))))/(len(set(y_test)))
    fscore_all = sum(2*(cm[i,i]/sum(cm[i,j] for j in range(len(set(y_test)))))*(cm[i,i]/sum(cm[j,i] for j in range(len(set(y_test)))))/(cm[i,i]/sum(cm[i,j] for j in range(len(set(y_test))))+cm[i,i]/sum(cm[j,i] for j in range(len(set(y_test))))) for i in range(len(set(y_test))))/len(set(y_test))
    
    TP = cm[1,1]
    FP = cm[0,1]
    TN = cm[0,0]
    FN = cm[1,0]
    # Precision for Positive = TP/(TP + FP)
    prec_pos = TP/(TP + FP)
    f1_pos = 2*TP/(TP*2 + FP+ FN)
    # TPR = TP/(TP+FN)
    TPR = cm[1,1]/sum(cm[1,j] for j in range(len(set(y_test))))
    # FPR = FP/(FP+TN)
    FPR = cm[0,1]/sum(cm[0,j] for j in range(len(set(y_test))))
    # specificity = TN/(FP+TN)
    Specificity = cm[0,0]/sum(cm[0,j] for j in range(len(set(y_test))))

    MCC = matthews_corrcoef(y_test, y_pred)

    CKappa = metrics.cohen_kappa_score(y_test, y_pred)

    # w_acc = (TP*20 + TN)/ [(TP+FN)*20 + (TN+FP)] if 20:1 ratio of non-feeding to feeding
    ratio = (TN+FP)/(TP+FN)

    w_acc = (TP*ratio + TN)/ ((TP+FN)*ratio + (TN+FP))

    # Show confusion matrix in a separate window
#     plt.matshow(cm)
#     plt.title('Confusion matrix')
#     plt.colorbar()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.show()
    
#     print(accuracy, recall_all, precision_all, fscore_all)
    return prec_pos, f1_pos, TPR, FPR, Specificity, MCC, CKappa, w_acc, cm




def poly_fit(signal,order):
    signal = np.ravel(signal)
    x = range(len(signal))
    x = np.linspace(x[0], x[-1], num=len(x))
    coefs = poly.polyfit(x, signal, order)
#     x_new = range(len(x))
#     x_new = np.linspace(x_new[0], x_new[-1], num=len(x_new))
#     ffit = poly.polyval(x_new, coefs)
#     plt.plot(x_new, ffit)
    return coefs




# Set the frame and step size
frame_size_seconds = 6
step_size_seconds = int(frame_size_seconds/2)
sampling_rate = 25
number_of_feat_per_axis = 11

frame_size = frame_size_seconds * sampling_rate
step_size = step_size_seconds * sampling_rate

# -----------------------------------------------------------------------------------
#
#
#
#
#
#		Train Model
#
#
#
#
#
#
# -----------------------------------------------------------------------------------
print ""
print "---------------------------------------------------------"
print ""
print ""
print ""
print ""
print "Train Model for All"
print ""
print ""
print ""
print ""
print ""
print "---------------------------------------------------------"
print ""

first_time_in_exclude_loop = 1
for p_counter in xrange(1, 21, 1):

	if p_counter==14:
		continue
		
	try:
		print "Loading: " + "../participants/" + str(p_counter) + "/datafiles/waccel_tc_ss_label.csv"
		L_T = genfromtxt("../participants/" + str(p_counter) + "/datafiles/waccel_tc_ss_label.csv", delimiter=',')
	except:
		error_participant_string = str(p_counter)
		print "** Error loading data for participant: " + str(p_counter)
		continue

	# Remove the relative timestamp
	L_T = L_T[:,1:]

	if first_time_in_exclude_loop==1:
		first_time_in_exclude_loop = 0
		Z_T = L_T
	else:
		Z_T = vstack((Z_T,L_T))

print ""
print "Shape of training data: " + str(Z_T.shape)
print ""
print str(Z_T)
print ""

# Number of inputs
number_of_inputs = Z_T.shape[1]-1

# -----------------------------------------------------------------------------------
#
#Training
#
# -----------------------------------------------------------------------------------

print ""
print "---------------------------------------------------------"
print " Loading Features + Build Model"
print "---------------------------------------------------------"
print ""

pos_examples_counter = 0
neg_examples_counter = 0

# Calculate features for frame
for counter in xrange(0,len(Z_T),step_size):

	# Add up labels
	A_T = Z_T[counter:counter+frame_size, number_of_inputs]
	S_T = sum(A_T)

	if S_T>step_size:
		pos_examples_counter = pos_examples_counter + 1
		S_T = 1
	else:
		neg_examples_counter = neg_examples_counter + 1
		S_T = 0

	R_T = Z_T[counter:counter+frame_size, :number_of_inputs] 

	M_T = mean(R_T,axis=0)
	V_T = var(R_T,axis=0)
	SK_T = stats.skew(R_T,axis=0)
	K_T = stats.kurtosis(R_T,axis=0)
	RMS_T = sqrt(mean(R_T**2,axis=0))
	MED = median(R_T,axis=0)
	MAX = R_T.max(axis=0)
	MIN = R_T.min(axis=0)
	Q3 = np.percentile(R_T,75,axis=0)
	Q1 = np.percentile(R_T,25,axis=0)
	COV_M = np.cov(R_T.T)
	COV = np.array([COV_M[0,1], COV_M[1,2], COV_M[0,2]])
	# COEFSX = poly_fit(R_T[:,0], 4)[:3]
	# COEFSY = poly_fit(R_T[:,1], 4)[:3]
	# COEFSZ = poly_fit(R_T[:,2], 4)[:3]


	H_T = hstack((M_T,V_T))
	H_T = hstack((H_T,SK_T))
	H_T = hstack((H_T,K_T))
	H_T = hstack((H_T,RMS_T))
	H_T = hstack((H_T,MED))
	H_T = hstack((H_T,MAX))
	H_T = hstack((H_T,MIN))
	H_T = hstack((H_T,Q3))
	H_T = hstack((H_T,Q1))
	H_T = hstack((H_T,COV))
	# H_T = hstack((H_T,COEFSX))
	# H_T = hstack((H_T,COEFSY))
	# H_T = hstack((H_T,COEFSZ))


	# ----------------------------- Label -------------------------------------

	# Add label
	H_T = hstack((H_T,S_T))
	if counter==0:
		F_T = H_T
	else:
		F_T = vstack((F_T,H_T))
		# if S_T==1:
		# 	for p_counter in xrange(0,5,1):
		# 		F_T = vstack((F_T,H_T))

print ""
print "Positive Examples: " + str(pos_examples_counter)
print "Negative Examples: " + str(neg_examples_counter)
print ""


print ""
print "Print of F_T: " + str(F_T)
print ""

# Get features and labels
X_T = F_T[:,:number_of_inputs*number_of_feat_per_axis]
Y_T = F_T[:,number_of_inputs*number_of_feat_per_axis]

print ""
print "X_T: " + str(X_T)

print ""
print "Shape of X_T: " +str(X_T.shape)

print ""
print "Y_T: " + str(Y_T)

print ""
print "Shape of Y_T: " + str(Y_T.shape)

# Train classifier
#clf = ExtraTreesClassifier(n_estimators=100)
clf = RandomForestClassifier(n_estimators=185)
#clf = AdaBoostClassifier(n_estimators=185)
#clf = KNeighborsClassifier(n_neighbors=2)
#clf = svm.LinearSVC()
#clf = GaussianNB()
#clf = DecisionTreeClassifier()
#clf = LogisticRegression()

print ""
print "Training model..."
clf.fit(X_T,Y_T)

print ""
print "Saving model..."
joblib.dump(clf, '../model/wrist.pkl')





# -----------------------------------------------------------------------------------
#
#
#
#
#
#		Test Model
#
#
#
#
#
#
# -----------------------------------------------------------------------------------
print ""
print "---------------------------------------------------------"
print ""
print ""
print ""
print ""
print "Generate test set features"
print ""
print ""
print "---------------------------------------------------------"
print ""

ts = time.time()


first_time_in_exclude_loop = 1
for p_counter in xrange(21, 22, 1):

	if p_counter==14:
		continue
		
	try:
		print "Loading: " + "../participants/" + str(p_counter) + "/datafiles/waccel_tc_ss_label.csv"
		L_T = genfromtxt("../participants/" + str(p_counter) + "/datafiles/waccel_tc_ss_label.csv", delimiter=',')
	except:
		error_participant_string = str(p_counter)
		print "** Error loading data for participant: " + str(p_counter)
		continue

	# Remove the relative timestamp
	L_T = L_T[:,1:]

	if first_time_in_exclude_loop==1:
		first_time_in_exclude_loop = 0
		Z_T = L_T
	else:
		Z_T = vstack((Z_T,L_T))

print ""
print "Shape of training data: " + str(Z_T.shape)
print ""
print str(Z_T)
print ""

# Number of inputs
number_of_inputs = Z_T.shape[1]-1

# -----------------------------------------------------------------------------------
#
#Testing
#
# -----------------------------------------------------------------------------------

print ""
print "---------------------------------------------------------"
print " Loading Features + Test Model"
print "---------------------------------------------------------"
print ""

pos_examples_counter = 0
neg_examples_counter = 0

# Calculate features for frame
for counter in xrange(0,len(Z_T),step_size):

	# Add up labels
	A_T = Z_T[counter:counter+frame_size, number_of_inputs]
	S_T = sum(A_T)

	if S_T>step_size:
		pos_examples_counter = pos_examples_counter + 1
		S_T = 1
	else:
		neg_examples_counter = neg_examples_counter + 1
		S_T = 0

	R_T = Z_T[counter:counter+frame_size, :number_of_inputs]

	# print(R_T)
	# print(R_T[:,0])

	M_T = mean(R_T,axis=0)
	V_T = var(R_T,axis=0)
	SK_T = stats.skew(R_T,axis=0)
	K_T = stats.kurtosis(R_T,axis=0)
	RMS_T = sqrt(mean(R_T**2,axis=0))
	MED = median(R_T,axis=0)
	MAX = R_T.max(axis=0)
	MIN = R_T.min(axis=0)
	Q3 = np.percentile(R_T, 75,axis=0)
	Q1 = np.percentile(R_T, 25,axis=0)
	COV_M = np.cov(R_T.T)
	COV = np.array([COV_M[0,1], COV_M[1,2], COV_M[0,2]])
	# COEFSX = poly_fit(R_T[:,0], 4)[:3]
	# COEFSY = poly_fit(R_T[:,1], 4)[:3]
	# COEFSZ = poly_fit(R_T[:,2], 4)[:3]




	H_T = hstack((M_T,V_T))
	H_T = hstack((H_T,SK_T))
	H_T = hstack((H_T,K_T))
	H_T = hstack((H_T,RMS_T))
	H_T = hstack((H_T,MED))
	H_T = hstack((H_T,MAX))
	H_T = hstack((H_T,MIN))
	H_T = hstack((H_T,Q3))
	H_T = hstack((H_T,Q1))
	H_T = hstack((H_T,COV))
	# H_T = hstack((H_T,COEFSX))
	# H_T = hstack((H_T,COEFSY))
	# H_T = hstack((H_T,COEFSZ))

	# ----------------------------- Label -------------------------------------

	# Add label
	H_T = hstack((H_T,S_T))
	if counter==0:
		F_T = H_T
	else:
		F_T = vstack((F_T,H_T))
		# if S_T==1:
		# 	for p_counter in xrange(0,5,1):
		# 		F_T = vstack((F_T,H_T))

print ""
print "Positive Examples: " + str(pos_examples_counter)
print "Negative Examples: " + str(neg_examples_counter)
print ""


print ""
print "Print of F_T: " + str(F_T)
print ""

# Get features and labels
X_T = F_T[:,:number_of_inputs*number_of_feat_per_axis]
Y_T = F_T[:,number_of_inputs*number_of_feat_per_axis]

print ""
print "X_T: " + str(X_T)

print ""
print "Shape of X_T: " +str(X_T.shape)

print ""
print "Y_T: " + str(Y_T)

print ""
print "Shape of Y_T: " + str(Y_T.shape)



predicted = clf.predict(X_T)


end = time.time()
print ""
print ""
print ""
print("time eclaped:")
print(end - ts)
print ""

print ""
print "Shape of Predicted: " + str(predicted.shape)
print ""

prec_pos, f1_pos, TPR, FPR, Specificity, MCC, CKappa, w_acc, cm = calc_cm(Y_T, predicted)
print(cm)
print(prec_pos)
print(f1_pos)



		