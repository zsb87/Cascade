import os
import re
import csv
import sys
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import time
import random
import glob
from sklearn import preprocessing
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
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
import _pickle as cPickle
from stru_utils_v2 import *


save_flg = 1

subjsIS = ['testP3','testP4','testP5','testP10','trainP3','trainP4','trainP5','trainP10']
subjsUS = ['testP1','testP3','testP6','testP7','testP8','trainP1','trainP3','trainP6','trainP7','trainP8'] 
subjsField = ['P3','P4','P5','P7','P10','P11','P12','P13','P14','P17','P18','P23','P24'] 

subjsProt = {'IS':subjsIS, 'US':subjsUS, 'Field':subjsField}
pathProt = {'IS':'IS', 'US':'US', 'Field':'Field/testdata_labeled'}

columns = ['Angular_Velocity_x','Angular_Velocity_y','Angular_Velocity_z','Linear_Accel_x','Linear_Accel_y','Linear_Accel_z']



def genFeedingFeats(protocol):
    
    allfeatDF = pd.DataFrame()

    for subj in subjsProt[protocol]:

        fgFolder = '../WillSense/code/WS/'+pathProt[protocol]+'/'+subj+'/feeding_gestures/'
        csvFiles = glob.glob(fgFolder + "*.csv")
        csvFilesS=sorted(csvFiles)

        for file in csvFilesS:
            r_df_gesture = pd.read_csv(file)
            r_df_gesture = r_df_gesture[columns]
            # print(r_df_gesture)
            feat = gen_feat_reduced(r_df_gesture)
            # print(feat)

            featDF = pd.DataFrame(feat[1:] , columns=feat[0])
            allfeatDF = pd.concat([allfeatDF,featDF])

    return allfeatDF


def genNFeedingFeats(protocol):
    
    allfeatDF = pd.DataFrame()

    for subj in subjsProt[protocol]:
        nfgFolder = '../WillSense/code/WS/'+pathProt[protocol]+'/'+subj+'/nonFeedingGestures/'

        csvFiles = glob.glob(nfgFolder + "*.csv")
        csvFilesS=sorted(csvFiles)

        for file in csvFilesS:
            r_df_gesture = pd.read_csv(file)
            r_df_gesture = r_df_gesture[columns]
            # print(r_df_gesture)
            feat = gen_feat_reduced(r_df_gesture)
            # print(feat)

            featDF = pd.DataFrame(feat[1:] , columns=feat[0])
            allfeatDF = pd.concat([allfeatDF,featDF])

    return allfeatDF

            

    # allfeatDF = pd.DataFrame()

    # for i in range(len(gt_headtail)):
    #     # saveFilePath = './WS/Field/gest_pred_data/pred_gesture_' + str(i) + '.csv'

    #     dataStart = int(gt_headtail['Start'].iloc[i])*2 - 2
    #     dataEnd = int(gt_headtail['End'].iloc[i])*2 + 1

    #     r_df_gesture = r_df.iloc[dataStart:dataEnd]
    #     r_df_gesture = r_df_gesture[['Angular_Velocity_x', 'Angular_Velocity_y', 'Angular_Velocity_z', 'Linear_Accel_x','Linear_Accel_y','Linear_Accel_z']]
                
    #     # r_df_gesture.to_csv(saveFilePath)
    #     if i % 500 == 0:
    #         print(i)

    #     # r_df_gesture = add_pitch_roll(r_df_gesture)

    #     # generate the features
    #     # feat = gen_feat(r_df_gesture)
    #     feat = gen_feat_reduced(r_df_gesture)

    #     featDF = pd.DataFrame(feat[1:] , columns=feat[0])
    #     allfeatDF = pd.concat([allfeatDF,featDF])


    # outfile = featFolder + subjs[p_counter] + "_run"+ str(run) +"_pred_features.csv"
    # allfeatDF.to_csv(outfile, index =None)

protocol = 'US'


fgAllfeatDF = genFeedingFeats(protocol)

fgFeatFolder = '../WillSense/code/WS/'+pathProt[protocol]+'/feature/feedingGestures/'
if not os.path.exists(fgFeatFolder):
    os.makedirs(fgFeatFolder)
outfile = fgFeatFolder + "features.csv"

fgAllfeatDF.to_csv(outfile, index =None)




nfgAllfeatDF = genNFeedingFeats(protocol)
nfgFeatFolder = '../WillSense/code/WS/'+pathProt[protocol]+'/feature/nonFeedingGestures/'
if not os.path.exists(nfgFeatFolder):
    os.makedirs(nfgFeatFolder)

outfile = nfgFeatFolder + "features.csv"

nfgAllfeatDF.to_csv(outfile, index =None)










