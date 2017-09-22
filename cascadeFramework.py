
# coding: utf-8

# In[8]:

import numpy as np
import pandas as pd
import os
get_ipython().magic('matplotlib inline')
import matplotlib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import pickle

from stru_utils_v2 import *
from cascadeUtil import weval, wUpdate, rmFilesInFolder, predWeakClf, saveWeakClf, loadWeakClf
from cascadeUtil import loadDataset, loadTrnTestDataset, XYTrnUpdateWithTopNFeats, updateTestSetbySelPosSamples, updateTrnsetWithFPtrueSamples
from cascadeUtil import buildStrongClfDefThres, loadStrongClfDefThres,loadStrongClfAdjThres


# # building a cascaded detector
# 
# # with feeding dataset

# In[9]:

featList = [23,24,21,28,8,3,4,1,7,14,27,11,13,26,6,17,18,22,2,29,16,25,9,5,30,20,19,10,12,15]
print(len(featList))


# In[15]:

datasets = ['feeding' , 'cancer']
dataset = datasets[0]
print('dataset:', dataset)

XY = loadDataset(dataset)

print(XY)


# initialization
XYPos =  XY[np.where(XY[:,-1]==1)[0],:]
XYNeg =  XY[np.where(XY[:,-1]==0)[0],:]

print('XYPos:',XYPos.shape)

XYPosTrn, XYPosTest = tt_split(XYPos, 0.3)
XYNegTrn, XYNegTest = tt_split(XYNeg, 0.3)

P = XYPosTrn
N = XYNegTrn

XYTrn = np.vstack((P,N))
print('XYTrn shape:', XYTrn.shape)
XYTest = np.vstack((XYPosTest,XYNegTest))


T = 100

f = 0.5
d = 0.95

FTar = 0.1

FList = []

F = 1 #F0
FPrev = 0
D = 1 #D0
DPrev = 0

clfThresList = []

# stage
i = 0 

if i == 0: # build the first stage
    nFeats = 0
    FPrev = F

# while F > FTar:
#     i = i + 1
#     nFeats = 0
#     FList.append(F)
#     FPrev = F    
#     DPrev = D
    
    print("Build stage 1:")


    while F > FPrev*f:
        
        nFeats = nFeats + 1
        print('nfeats:', nFeats)
        
        ######################################################################################
        #       build strong classifier with only first nFeats features in train set 
        ######################################################################################
        
        mdlpath = './fd_model_stage'+str(i)+'/'
    
        XYTrnNFeat = XYTrnUpdateWithTopNFeats(XYTrn, featList, nFeats)

#       build stage with features selected
        buildStrongClfDefThres(XYTrnNFeat, T, mdlpath)
    
#       evaluate cascaded classifier on validation set to determine F and D
        yRes, clfThres = loadStrongClfDefThres(XYTest, T, mdlpath)
        
        print("clfThres: ", clfThres)
        prec_pos, D, f1_pos, TPR, F, Specificity, MCC, CKappa, w_acc, cm = calc_cm_rcall(XYTest[:,-1], yRes)
        print("\nrecall_pos: ",D)
        print("false positive rate: ", F)
        
        if D > d and F < f:
            clfThresList.append(clfThres)
            FList.append(F)
            
        elif D < d:
            thres = int(clfThres)
            
            while thres > 2 and D < d:
                
                thres = thres - 2
                yRes = loadStrongClfAdjThres(XYTrnNFeat, T, mdlpath , thres)
                print("thres: ", thres)

                prec_pos, D, f1_pos, TPR, F, Specificity, MCC, CKappa, w_acc, cm = calc_cm_rcall(XYTrnNFeat[:,-1], yRes)

                print("recall_pos: ",D)
                print("false positive rate: ", F)
                

            if D > d:
                if F < f:
                    clfThresList.append(thres)
                    FList.append(F)
                    print('Succeed!', '\n')
                else:
                    print('Fail: F cannot be less than f when D is greater than or equal to d', '\n')
            else:
                print('Fail, D cannot be greater than or equal to d', '\n')
            
            

        elif F > f:
                print('Specific case pursued came out: F > f !!! need to tune F, D, f, d', '\n')
                
#       prediction for 
        
#       evalutation

#     update sample set
    XYTrn = updateTrnsetWithFPtrueSamples(XYTrn, yRes)



# In[12]:

FList


# In[16]:

clfThresList


# In[17]:

XYTrn.shape


# In[ ]:

i = 1

if i == 1: # build the 2nd stage
    FPrev = FList[0]
    F = FPrev
    print("Build stage 2:")

    while F > FPrev*f:
        nFeats = nFeats + 1
        print('nFeats:', nFeats)
        
        ######################################################################################
        #       build strong classifier with only first nFeats features in train set 
        ######################################################################################
        
        mdlpath = './fd_model_stage'+str(i)+'/'
        
        XYTrnNFeat = XYTrnUpdateWithTopNFeats(XYTrn, featList, nFeats)# XYTrn is updated

#       build stage with features selected
        buildStrongClfDefThres(XYTrnNFeat, T, mdlpath)
    
#       evaluate cascaded classifier on validation set to determine F and D
        yRes = loadStrongClfAdjThres(XYTest, T, mdlpath, clfThresList[0])
        
        print("clfThres: ", clfThres)
        prec_pos, D, f1_pos, TPR, F, Specificity, MCC, CKappa, w_acc, cm = calc_cm_rcall(XYTest[:,-1], yRes)
        print("\nrecall_pos: ",D)
        print("false positive rate: ", F)
        
        if D > d and F < f:
            clfThresList.append(clfThres)
            FList.append(F)
            
        elif D < d:
            thres = int(clfThres)
            
            while thres > 2 and D < d:
                
                thres = thres - 2
                yRes = loadStrongClfAdjThres(XYTrnNFeat, T, mdlpath , thres)
                print("thres: ", thres)

                prec_pos, D, f1_pos, TPR, F, Specificity, MCC, CKappa, w_acc, cm = calc_cm_rcall(XYTrnNFeat[:,-1], yRes)

                print("recall_pos: ",D)
                print("false positive rate: ", F)
                

            if D > d:
                if F < f:
                    clfThresList.append(thres)
                    FList.append(F)
                    print('Succeed!', '\n')
                else:
                    print('Fail: F cannot be less than f when D is greater than or equal to d', '\n')
            else:
                print('Fail, D cannot be greater than or equal to d', '\n')
            
            

        elif F > f:
                print('Specific case pursued came out: F > f !!! need to tune F, D, f, d', '\n')
                
#       prediction for 
        
#       evalutation



#     update sample set
#     XYTrn = XYTrnUpdate(XYTrn, yRes, featList, nFeats, thres)



# In[ ]:



