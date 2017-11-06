
# coding: utf-8

# In[5]:

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
from sklearn.preprocessing import normalize
import pickle
from stru_utils_v2 import *


# In[42]:

def rmFilesInFolder(folder):
#     remove files in folder
    import os, shutil
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path): 
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)


# # decision stump scikit-learn wrapper

# In[43]:

def predWeakClf(X,y,d=None):
# eg: predWeakClf(xyFNFArr[:,f],xyFNFArr[:,-1])
# input type:
#     X: ndarray
#     y: ndarray
# output type:
#     pred: ndarray

    h = DecisionTreeClassifier(max_depth=1)
    X = X.reshape(-1, 1)
    h.fit(X, y, sample_weight=d)
    pred = h.predict(X)

    return pred



# mdl saves the modified weak classifier model, 
# featFile saves which feature each weak classifier is using

def saveWeakClf(X, y, mdlname, feat, beta, configFile, d=None):
#     X is one single feature
#     feat is the index of this input feature
#     beta is 
    h = DecisionTreeClassifier(max_depth=1)
    X = X.reshape(-1, 1)
    h.fit(X, y, sample_weight=d)
    
    # save the model to disk
    pickle.dump(h, open(mdlname, 'wb'))
    
    with open(configFile, 'w') as text_file:
        text_file.write(str(feat))   
        text_file.write('\n')   
        text_file.write(str(beta))






# mdl saves the modified weak classifier model, 
# configFile saves which feature each weak classifier is using and the beta for each classifier

def loadWeakClf(mdlname, configFile):
    # load the model from disk
    loaded_model = pickle.load(open(mdlname, 'rb'))
    
    with open(configFile, "r") as f:
        array = []
        for line in f:
            array.append(line)
            
    feat = array[0]
    beta = array[1]
    
    return loaded_model, feat, beta


# test case
# loaded_model, feat, beta = (loadWeakClf('./model/0.sav', './model/0_feat.sav'))
# print(feat)
# print(beta)


# ## boosting component

# In[2]:

def weval(yLabel,yPred,w_norm):
#     
#     calculate error
# 
# input type:
#     yLabel: ndarray
#     yPred: ndarray
#     w_norm: ndarray

    return np.dot(np.absolute(yLabel - yPred), w_norm)

'''
################################# test case #############################
'''
# weval(np.asarray([0,1,0,1,1]),np.asarray([1,1,1,0,1]),np.asarray([0.2,0.1,0.3,.2,.2]))


# In[6]:

def wUpdate(beta, t, errmin, w_norm, samples, yLabel, yPred):
# 
# update weights and betas
# 
    beta[t] = errmin/(1-errmin)
    
    w = np.multiply(w_norm, np.power(beta[t], np.ones(samples) - np.absolute(yLabel - yPred)))
    return w, beta

'''
################################# test case #############################
## step by step manual result for this test code
##
## np.absolute(yLabel - yPred) = [0,1,0]
## np.ones(samples) - np.absolute(yLabel - yPred) = [1,0,1]
## np.power(beta[t], np.ones(samples) - np.absolute(yLabel - yPred)) = [0.1111,1,0.1111]
## w = [ 0.02222222,  0.4       ,  0.04444444]
##
#########################################################################
'''
wUpdate(beta=np.zeros(10), t=1, errmin=0.1, w_norm=np.array([0.2,0.4,0.4]), samples=3, yLabel=np.array([0,1,0]), yPred=np.array([0,0,0]))


# # import dataset

# In[46]:

def loadDataset(dataset):


    if dataset == 'feeding':

        protocol = 'IS'

        pathProt = {'IS':'IS', 'US':'US', 'Field':'Field/testdata_labeled'}

        fgFeatFolder = '../WillSense/code/WS/'+pathProt[protocol]+'/feature/feedingGestures/'
        file = fgFeatFolder + "features.csv"
        fgDf = pd.read_csv(file)
        fgDf = fgDf.dropna()

        nfgFeatFolder = '../WillSense/code/WS/'+pathProt[protocol]+'/feature/nonFeedingGestures/'
        file = nfgFeatFolder + "features.csv"
        nfgDf = pd.read_csv(file)
        nfgDf = nfgDf.dropna()


        # Feeding
        fgFeatsArr = fgDf.as_matrix()
        xyFgArr = np.hstack((fgFeatsArr, np.ones((fgFeatsArr.shape[0],1))))

        # Non-feeding
        nfgFeatsArr = nfgDf.as_matrix()

        balancePN = 1

        if balancePN:
        # balance pos and neg samples, here #pos < #neg:
            nfgFeatsArr = nfgFeatsArr[:fgFeatsArr.shape[0],:]
            xyNfgArr = np.hstack((nfgFeatsArr, np.zeros((nfgFeatsArr.shape[0],1))))
        else:
            xyNfgArr = np.hstack((nfgFeatsArr, np.zeros((nfgFeatsArr.shape[0],1))))


        xyFNFArr = np.vstack((xyFgArr,xyNfgArr))

        np.random.shuffle(xyFNFArr)#no return function


        XY = xyFNFArr


    else:

        data = load_breast_cancer(return_X_y=True)

        X_T = data[0]
        Y_T = data[1]

        XY = np.hstack((X_T,Y_T.reshape(-1, 1)))
    #     np.savetxt('breast_cancer.csv', XY, delimiter=',')
    
    return XY


# In[47]:

def loadTrnTestDataset(dataset):


    XY = loadDataset(dataset)
    
    XYPos =  XY[np.where(XY[:,-1]==1)[0],:]
    XYNeg =  XY[np.where(XY[:,-1]==0)[0],:]

    XYPosTrn, XYPosTest = tt_split(XYPos, 0.3)
    XYNegTrn, XYNegTest = tt_split(XYNeg, 0.3)

    XYTrn = np.vstack((XYPosTrn,XYNegTrn))
    XYTest = np.vstack((XYPosTest,XYNegTest))
    
    return XYTrn, XYTest


# In[6]:

def XYTrnUpdateWithTopNFeats(XYTrn, featList, nFeats):
    
#       select first n features in featList for training set
    feats = featList[:nFeats]
    XYTrnNFeat = np.hstack((XYTrn[:,feats], XYTrn[:,-1].reshape([-1,1])))

    return XYTrnNFeat


# In[7]:

def updateTestSetbySelPosSamples(XYTestNFeat, yRes):
    
    indList = []
    for i in range(len(yRes)):
        if yRes[i] == 1:
            indList.append(i)

    print(indList)
    XYTestNFeat_1 = XYTestNFeat[indList,:]
    
    return XYTestNFeat_1


# In[ ]:

def updateTrnsetWithFPtrueSamples(XYCurrTrn, yRes):
    yLabel = XYCurrTrn[:,-1]
    indList = []
    
    for i in range(len(yRes)):
        if yRes[i] == 1 and yLabel[i]==0:
            indList.append(i)
            
    N = XYCurrTrn[indList,:]
    P = XYCurrTrn[np.where(XYCurrTrn[:,-1]==1)[0],:]

    XYTrn = np.vstack((P,N))

    return XYTrn
    


# In[48]:

# np.savetxt('w_norm_rec.out', w_norm_rec, delimiter=',')


# In[49]:

# print(w)


# In[50]:

# print(fOptList)


# In[51]:

# yPred = predWeakClf(XY[:,4].reshape(-1, 1), yLabel, w_norm)
# print(weval(yLabel, yPred, w_norm))
# print(yPred-yLabel)


# In[52]:

# for i in fOptList:
#     print(list(fgDf)[i])


# In[ ]:




# ## Module for Cascaded Classifier

# In[53]:

def buildStrongClfDefThres(XY, T , mdlpath = './mdl/'):
    
    from sklearn import preprocessing


# input:
#           XY:   features and labels
#            T:   number of iterations, IMPORTANT PARAMETER
#      mdlpath:   the path where the models are saved
# 
# output:
#   saveFinish:   1 for success, 0 for failure
# 
# save:
#        model
#         fOpt:   feature index
#         beta
# 

    saveFinish = 0
    
    yLabel = XY[:,-1]
    
#     featValid = featList[:nFeats]
#     X = XY[:,featValid]
#     XY = np.hstack((X,XY[:,-1]))

    nsamples = list(XY[:,-1]).count(0)
    psamples = list(XY[:,-1]).count(1)
    samples = nsamples + psamples
    
    # number of features
    nFeats = XY.shape[1]-1
    
    featList = list(range(nFeats))

    if not os.path.exists(mdlpath):
        os.makedirs(mdlpath)
    else:
        rmFilesInFolder(mdlpath)
        
    fOptList = []
    
    betas = np.zeros([T])# keep record of all betas in all rounds
#     w_norm_rec = np.zeros([T, samples])

    # initialize weights
    w = np.zeros([samples])
    w[:nsamples] = 1/nsamples
    w[nsamples:] = 1/psamples


    for t in range(T):
        # 1. normalize weights
        w_norm = preprocessing.normalize(w, norm = 'l1')
#         w_norm_rec[t,:] = w_norm # w_norm record

        err = np.ones(nFeats)

        # for each feature:
        for f in featList:
            # 2. train a classifier h using a single feature.
            w_norm = w_norm.ravel()        

            yPred = predWeakClf(XY[:,f].reshape(-1, 1), yLabel, w_norm)

            # error calculation
            err[f] = weval(yLabel, yPred, w_norm)

        # 3. choose the classifier with lowest error
        fOpt = np.argmin(err)

#         if t%(T/10) == 0:
#             print('fOpt: ',fOpt)

        errmin = np.amin(err)    
        
#         if t%(T/10) == 0:
#             print(errmin)

        fOptList.append(fOpt)


        # 4. update weights
        w, betas = wUpdate(betas, t, errmin, w_norm, samples, yLabel, yPred)

        saveWeakClf(XY[:,np.argmin(err)].reshape(-1, 1), yLabel, 
                       mdlpath+str(t)+'.sav', 
                       fOpt, betas[t], mdlpath+str(t)+'_feat.sav', 
                       w_norm)
        
#         if t%(T/10) == 0:
#             print('t:',t)

            
    saveFinish = 1


    return saveFinish


# In[54]:

'''
# After training with training set.
# Load weak classifier and then build final strong classifier with default threshold, 
# meaning threshold is not adjustable.
# Note: fOptList is basically the weak classifier h(x).
'''

# mdlpath = './model/'
# the models are './model/(i).sav'

def loadStrongClfDefThres(XY, T, mdlpath):
    
#     load and get weak classifiers result

    yPredRec = np.zeros([T, XY.shape[0]])
    yRes = np.zeros(XY.shape[0])
    X = XY[:,:-1]
    betas = np.zeros([T])# keep record of all betas in all rounds

    for t in range(T):
        h, feat, betas[t] = loadWeakClf(mdlpath+str(t)+'.sav', mdlpath+str(t)+'_feat.sav')
        yPred = h.predict(X[:,feat].reshape(-1, 1))
        yPredRec[t,:] = yPred
        
#     calc classify threshold

    betas_recip = np.reciprocal(betas)
    alphas = np.log(betas_recip)
    clfThres = np.sum(alphas)*0.5
    
    yComb = np.dot(alphas, yPredRec)
    
#     get final result

    for i in range(yComb.shape[0]):
        if yComb[i] < clfThres:
            yRes[i] = 0
        else:
            yRes[i] = 1
    
    return yRes, clfThres


# In[55]:

'''
# final strong classifier with adjustable threshold
'''    

# mdlpath = './model/'
# the models are './model/(i).sav'


def loadStrongClfAdjThres(XY, T, mdlpath ,clfThres):

#     load and get weak classifiers result

    yPredRec = np.zeros([T, XY.shape[0]])
    yRes = np.zeros(XY.shape[0])
    X = XY[:,:-1]
    betas = np.zeros([T])# keep record of all betas in all rounds

    for t in range(T):
        h, feat, betas[t] = loadWeakClf(mdlpath+str(t)+'.sav', mdlpath+str(t)+'_feat.sav')
        yPred = h.predict(X[:,feat].reshape(-1, 1))
        yPredRec[t,:] = yPred
        
#     calc classify threshold

    betas_recip = np.reciprocal(betas)
    alphas = np.log(betas_recip)
    
    yComb = np.dot(alphas, yPredRec) 
    
#     print(yComb)

    for i in range(yComb.shape[0]):
        if yComb[i] < clfThres:
            yRes[i] = 0
        else:
            yRes[i] = 1
    
    return yRes


# ## component for cascaded clf

# In[57]:

if __name__ == "__main__":
    
    datasets = ['feeding' , 'cancer']
    dataset = datasets[0]
    print('dataset:', dataset)

    XY = loadDataset(dataset)
    
    X_T = XY[:,:-1]
    Y_T = XY[:,-1]
    
#     XYTrn, XYTest = loadTrnTestDataset(dataset)
#     X_T = XYTrn[:,:-1]
#     Y_T = XYTrn[:,-1]
    
    ##########################################
    #     baseline model
    ##########################################

    baseModel = 1

    classifiers = ['KNN5', 'AdaBoost']

    if baseModel == 1:
        # Train classifier
        for classifier in classifiers:
            
            print('\nBaseline model:', classifier)

            if classifier == "KNN5":
                clf = KNeighborsClassifier(n_neighbors=5)
            elif classifier == "RF185":
                clf = RandomForestClassifier(n_estimators=185)
            elif classifier == "RF100":
                clf = RandomForestClassifier(n_estimators=100)
            elif classifier == "AdaBoost":
                clf = AdaBoostClassifier(n_estimators=100)

            #clf = ExtraTreesClassifier(n_estimators=100)
            #clf = AdaBoostClassifier(n_estimators=185)
            #clf = svm.LinearSVC()
            #clf = GaussianNB()
            #clf = DecisionTreeClassifier()
            #clf = LogisticRegression()

            clf.fit(X_T,Y_T)
            y_pred = clf.predict(X_T)

            prec_pos, recall_pos, f1_pos, TPR, FPR, Specificity, MCC, CKappa, w_acc, cm = calc_cm_rcall(Y_T, y_pred)

            
    

# '''
#     Strong classifier -- modified Adaboost
# '''

    from numpy import vectorize
    from sklearn.tree import *

    mdlpath = './model/'
    
    saveFinish = buildStrongClfDefThres(XY, 10, mdlpath)

    assert  saveFinish == 1

    yRes, clfThres = loadStrongClfDefThres(XY, 10, mdlpath)
    
    print("clfThres: ", clfThres,"\n")
    prec_pos, recall_pos, f1_pos, TPR, FPR, Specificity, MCC, CKappa, w_acc, cm = calc_cm_rcall(XY[:,-1], yRes)
    
    print("\nrecall_pos: ",recall_pos,'\n')
    print("prec_pos: ", prec_pos, '\n')
    print("false positive rate: ", FPR, '\n')
    
    for clfThres in range(1,20,5):
        
        yRes = loadStrongClfAdjThres(XY, 10, mdlpath ,clfThres)
        print("clfThres: ", clfThres)

        prec_pos, recall_pos, f1_pos, TPR, FPR, Specificity, MCC, CKappa, w_acc, cm = calc_cm_rcall(XY[:,-1], yRes)

        print("recall_pos: ",recall_pos)
        print("prec_pos: ", prec_pos)
        print("false positive rate: ", FPR, '\n')
        
        
        
        
    featList = [23,24,21,28,8,3,4,1,7,14,27,11,13,26,6,17,18,22,2,29,16,25,9,5,30,20,19,10,12,15]
    print(len(featList))
        
        
        
    f = 0.7
    d = 0.95

    FTar = 0.1

    FList = []
    clfThresList = []
    F = 1 #F0
    FPrev = 0
    D = 1 #D0
    DPrev = 0
    
    
    
    
    
    datasets = ['feeding' , 'cancer']
    dataset = datasets[1]
    print('dataset:', dataset)
    XY = loadDataset(dataset)

    # initialization
    XYPos =  XY[np.where(XY[:,-1]==1)[0],:]
    XYNeg =  XY[np.where(XY[:,-1]==0)[0],:]

    XYPosTrn, XYPosTest = tt_split(XYPos, 0.3)
    XYNegTrn, XYNegTest = tt_split(XYNeg, 0.3)

    P = XYPosTrn
    N = XYNegTrn


    XYTrn = np.vstack((XYPosTrn,XYNegTrn))
    XYTest = np.vstack((XYPosTest,XYNegTest))


    nFeats = 1
    feats = featList[:nFeats]
    XYTrnNFeat = np.hstack((XYTrn[:,feats], XYTrn[:,-1].reshape([-1,1])))
    XYTestNFeat = np.hstack((XYTest[:,feats], XYTest[:,-1].reshape([-1,1])))
    print('XYTestNFeat:')
    print(XYTestNFeat.shape)



    T = 100


    i = 0
    mdlpath = './model_stage'+str(i)+'/'
    saveFinish = buildStrongClfDefThres(XYTrnNFeat, T, mdlpath)
    assert saveFinish


    #       evaluate cascaded classifier on validation set to determine F and D
    yRes, clfThres = loadStrongClfDefThres(XYTestNFeat, T, mdlpath)

    print("clfThres: ", clfThres,"\n")
    prec_pos, D, f1_pos, TPR, F, Specificity, MCC, CKappa, w_acc, cm = calc_cm_rcall(XYTestNFeat[:,-1], yRes)
    print("\nrecall_pos: ",D,'\n')
    print("false positive rate: ", F, '\n')


    # # 
    # # plot to show ROC curve
    # # 
    # recalls = np.zeros(int(clfThres)+9)
    # FPRs = np.zeros(int(clfThres)+9)

    # i = 0
    # for clfThres in range(1,int(clfThres)+10):

    #     yRes = loadStrongClfAdjThres(XYTestNFeat, T, mdlpath ,clfThres)
    # #     print("clfThres: ", clfThres)
    #     prec_pos, recalls[i], f1_pos, TPR, FPRs[i], Specificity, MCC, CKappa, w_acc, cm = calc_cm_rcall(XYTestNFeat[:,-1], yRes)

    # #     print("false positive rate: ", FPR, '\n')
    #     i = i + 1

    # plt.figure(1)
    # plt.plot(FPRs, recalls)
    # plt.axvline(0.3, color='k', linestyle='--')
    # plt.axhline(0.95, color='k', linestyle='--')


    # featInd = []
    # for i in range(T):
    #     configFile = mdlpath + str(i) + '_feat.sav'
    #     with open(configFile, "r") as f:
    #         array = []
    #         for line in f:
    #             array.append(line)
    #         featInd.append(array[0])
    # print(set(featInd))


    # plt.xlabel('False positive rate')
    # plt.ylabel('True positive rate')
    # plt.title('ROC curve, # of features:'+ str(nFeats) +', rounds:'+str(T)+ ', feature:'+str(set(featInd)))



    # 
    # find the first acceptable threshold during decreasing threshold
    # 

    i = int(clfThres)
    F = 1 
    print("D:",D)

    while i > 1 and F > f and nFeats<30:

        print('F is greater than set f')

        while D < d:
            print('D is less than set d')

            clfThres = i - 1
            yRes = loadStrongClfAdjThres(XYTestNFeat, T, mdlpath ,clfThres)
            print("clfThres: ", clfThres)
            prec_pos, D, f1_pos, TPR, F, Specificity, MCC, CKappa, w_acc, cm = calc_cm_rcall(XYTestNFeat[:,-1], yRes)

            print("false positive rate: ", F, '\n')
            print("recall: ", D, '\n')

            i = i - 1

    clfThresList.append(clfThres)
    F1= F
    FList.append(F)


    yRes = loadStrongClfAdjThres(XYTestNFeat, T, './model_stage0/',clfThresList[0])
    prec_pos, D0, f1_pos, TPR, F0, Specificity, MCC, CKappa, w_acc, cm = calc_cm_rcall(XYTestNFeat[:,-1], yRes)

    print('D0:',D0)
    
    
    XYTrn = updateTrnUsingFP_allTrueSamples(XYPosTrn, yRes, XYTestNFeat)
    
    
    
    nFeats = 2
    feats = featList[:nFeats]
    #    generate new train set and keep old test set, for the second stage
    XYTrnNFeat = np.hstack((XYTrn[:,feats], XYTrn[:,-1].reshape([-1,1])))
    XYTestNFeat = np.hstack((XYTest[:,feats], XYTest[:,-1].reshape([-1,1])))


    T = 100

    # stage 2
    i = 1
    mdlpath = './model_stage'+str(i)+'/'
    saveFinish = buildStrongClfDefThres(XYTrnNFeat, T, mdlpath)
    assert saveFinish


    #       evaluate cascaded classifier on validation set to determine F and D
    # load stage 1 to test on test set


    XYTestNFeat0 = np.hstack((XYTestNFeat[:,:1].reshape(-1,1), XYTestNFeat[:,-1].reshape([-1,1])))
    print(XYTestNFeat0.shape)

    # load the tuned threshold
    yRes = loadStrongClfAdjThres(XYTestNFeat, T, './model_stage0/',clfThresList[0])
    prec_pos, D0, f1_pos, TPR, F0, Specificity, MCC, CKappa, w_acc, cm = calc_cm_rcall(XYTestNFeat[:,-1], yRes)

    yLabel = XYTestNFeat[:,-1]
    indList = []
    for i in range(len(yRes)):
        if yRes[i] == 1:
            indList.append(i)

    print(indList)
    XYTestNFeat_1 = XYTestNFeat[indList,:]

    yRes, clfThres = loadStrongClfDefThres(XYTestNFeat_1, T, './model_stage1/')
    prec_pos, D1, f1_pos, TPR, F1, Specificity, MCC, CKappa, w_acc, cm = calc_cm_rcall(XYTestNFeat_1[:,-1], yRes)

    print('Threshold: ', clfThres)
    F = F0*F1
    print('F0:',F0)
    print('F1:',F1)
    print('F:',F)
    print('\n')

    D = D0*D1
    print('D0:',D0)
    print('D1:',D1)
    print('D:',D)

    i = int(clfThres)


    while i > 1:

        i = i - 5

        yRes = loadStrongClfAdjThres(XYTestNFeat_1, T, './model_stage1/', i)
        prec_pos, D1, f1_pos, TPR, F1, Specificity, MCC, CKappa, w_acc, cm = calc_cm_rcall(XYTestNFeat_1[:,-1], yRes)


        print('Threshold: ', i)
        F = F0*F1
        print('F0:',F0)
        print('F1:',F1)
        print('F:',F)

        D = D0*D1
        print('D0:',D0)
        print('D1:',D1)
        print('D:',D)




        






    nFeats = 5
    feats = featList[:nFeats]
    #    generate new train set and keep old test set, for the second stage
    XYTrnNFeat = np.hstack((XYTrn[:,feats], XYTrn[:,-1].reshape([-1,1])))
    XYTestNFeat = np.hstack((XYTest[:,feats], XYTest[:,-1].reshape([-1,1])))


    T = 100

    # stage 2
    i = 1
    mdlpath = './model_stage'+str(i)+'/'
    saveFinish = buildStrongClfDefThres(XYTrnNFeat, T, mdlpath)
    assert saveFinish


    #       evaluate cascaded classifier on validation set to determine F and D
    # load stage 1 to test on test set


    XYTestNFeat0 = np.hstack((XYTestNFeat[:,:1].reshape(-1,1), XYTestNFeat[:,-1].reshape([-1,1])))
    print(XYTestNFeat0.shape)

    # load the tuned threshold
    yRes = loadStrongClfAdjThres(XYTestNFeat, T, './model_stage0/',clfThresList[0])
    prec_pos, D0, f1_pos, TPR, F0, Specificity, MCC, CKappa, w_acc, cm = calc_cm_rcall(XYTestNFeat[:,-1], yRes)


    # update the test set by selecting positive samples after prediction 
    XYTestNFeat_1 = updateTestSetbySelPosSamples(XYTestNFeat, yRes)


    yRes, clfThres = loadStrongClfDefThres(XYTestNFeat_1, T, './model_stage1/')
    prec_pos, D1, f1_pos, TPR, F1, Specificity, MCC, CKappa, w_acc, cm = calc_cm_rcall(XYTestNFeat_1[:,-1], yRes)

    print('Threshold: ', clfThres)
    F = F0*F1
    print('F0:',F0)
    print('F1:',F1)
    print('F:',F)
    print('\n')

    D = D0*D1
    print('D0:',D0)
    print('D1:',D1)
    print('D:',D)
    print('\n')


    if D1 > d and F1 < f:
        clfThresList.append(iThre)
        F1= F
        FList.append(F)
        print('Succeed!')

    else:

        iThre = int(clfThres)

        while iThre > 1 and D1 < d:
            iThre = iThre - 5

            yRes = loadStrongClfAdjThres(XYTestNFeat_1, T, './model_stage1/', iThre)
            prec_pos, D1, f1_pos, TPR, F1, Specificity, MCC, CKappa, w_acc, cm = calc_cm_rcall(XYTestNFeat_1[:,-1], yRes)


            print('Threshold: ', iThre)
            F = F0*F1
            print('F0:',F0)
            print('F1:',F1)
            print('F:',F)
            print('\n')


            D = D0*D1
            print('D0:',D0)
            print('D1:',D1)
            print('D:',D)
            print('\n')


        if D1 > d:
            if F1 < f:
                clfThresList.append(iThre)
                F1= F
                FList.append(F)
                print('Succeed!')
            else:
                print('Fail: F cannot be less than f when D is greater than or equal to d')
        else:
            print('Fail, D cannot be greater than or equal to d')



            
            
            
    print('overall FPRs in each step list: ', FList)
    yLabel = XYTestNFeat[:,-1]
    indList = []
    for i in range(len(yRes)):
        if yRes[i] == 1 and yLabel[i]==0:
            indList.append(i)
    print(indList)
    N = XYTest[indList,:]
    N_stage0 = N

    P = XYPosTrn
    XYTrn = np.vstack((P,N))
            


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# ## The first stage: set the threshold as 11,  D(recall) = .972, F(FPR) = .208
# ## number of features: 1, rounds = 100
# 
# ### as F>F(target) = 0.1, so put false detections into set N

# In[ ]:

# FList


# #### choose FP samples and all true samples as TRAIN SET

# In[1]:

def updateTrnUsingFP_allTrueSamples(XYPosTrn, yRes, XYTestNFeat):
    yLabel = XYTestNFeat[:,-1]
    indList = []

    for i in range(len(yRes)):
        if yRes[i] == 1 and yLabel[i]==0:
            indList.append(i)
    print(indList)
    N = XYTest[indList,:]
    N_stage0 = N

    P = XYPosTrn
    XYTrn = np.vstack((P,N))

    print(len(XYTrn))

    return XYTrn


# ## STAGE 2
# 
# ### Use P and updated N to train the second stage classifier, 
# ### Use test set and the whole cascaded clf(STAGE1&2) to test:
# 
# 
# 
# ### when #of feature = 2:
# 

# In[ ]:

# nFeats = 2
# feats = featList[:nFeats]
# #    generate new train set and keep old test set, for the second stage
# XYTrnNFeat = np.hstack((XYTrn[:,feats], XYTrn[:,-1].reshape([-1,1])))
# XYTestNFeat = np.hstack((XYTest[:,feats], XYTest[:,-1].reshape([-1,1])))


# T = 100

# # stage 2
# i = 1
# mdlpath = './model_stage'+str(i)+'/'
# saveFinish = buildStrongClfDefThres(XYTrnNFeat, T, mdlpath)
# assert saveFinish


# #       evaluate cascaded classifier on validation set to determine F and D
# # load stage 1 to test on test set


# XYTestNFeat0 = np.hstack((XYTestNFeat[:,:1].reshape(-1,1), XYTestNFeat[:,-1].reshape([-1,1])))
# print(XYTestNFeat0.shape)

# # load the tuned threshold
# yRes = loadStrongClfAdjThres(XYTestNFeat, T, './model_stage0/',clfThresList[0])
# prec_pos, D0, f1_pos, TPR, F0, Specificity, MCC, CKappa, w_acc, cm = calc_cm_rcall(XYTestNFeat[:,-1], yRes)

# yLabel = XYTestNFeat[:,-1]
# indList = []
# for i in range(len(yRes)):
#     if yRes[i] == 1:
#         indList.append(i)
        
# print(indList)
# XYTestNFeat_1 = XYTestNFeat[indList,:]

# yRes, clfThres = loadStrongClfDefThres(XYTestNFeat_1, T, './model_stage1/')
# prec_pos, D1, f1_pos, TPR, F1, Specificity, MCC, CKappa, w_acc, cm = calc_cm_rcall(XYTestNFeat_1[:,-1], yRes)

# print('Threshold: ', clfThres)
# F = F0*F1
# print('F0:',F0)
# print('F1:',F1)
# print('F:',F)
# print('\n')

# D = D0*D1
# print('D0:',D0)
# print('D1:',D1)
# print('D:',D)

# i = int(clfThres)


# while i > 1:
    
#     i = i - 5

#     yRes = loadStrongClfAdjThres(XYTestNFeat_1, T, './model_stage1/', i)
#     prec_pos, D1, f1_pos, TPR, F1, Specificity, MCC, CKappa, w_acc, cm = calc_cm_rcall(XYTestNFeat_1[:,-1], yRes)


#     print('Threshold: ', i)
#     F = F0*F1
#     print('F0:',F0)
#     print('F1:',F1)
#     print('F:',F)

#     D = D0*D1
#     print('D0:',D0)
#     print('D1:',D1)
#     print('D:',D)




# ## So,  when # of features = 2, cannot satisfy requirement of f = 0.3 and d = 0.95

# ### same with # of features = 3
# ## retry STAGE 2
# 
# ### when #of feature = 5:

# In[ ]:

#     nFeats = 5
#     feats = featList[:nFeats]
#     #    generate new train set and keep old test set, for the second stage
#     XYTrnNFeat = np.hstack((XYTrn[:,feats], XYTrn[:,-1].reshape([-1,1])))
#     XYTestNFeat = np.hstack((XYTest[:,feats], XYTest[:,-1].reshape([-1,1])))


#     T = 100

#     # stage 2
#     i = 1
#     mdlpath = './model_stage'+str(i)+'/'
#     saveFinish = buildStrongClfDefThres(XYTrnNFeat, T, mdlpath)
#     assert saveFinish


#     #       evaluate cascaded classifier on validation set to determine F and D
#     # load stage 1 to test on test set


#     XYTestNFeat0 = np.hstack((XYTestNFeat[:,:1].reshape(-1,1), XYTestNFeat[:,-1].reshape([-1,1])))
#     print(XYTestNFeat0.shape)

#     # load the tuned threshold
#     yRes = loadStrongClfAdjThres(XYTestNFeat, T, './model_stage0/',clfThresList[0])
#     prec_pos, D0, f1_pos, TPR, F0, Specificity, MCC, CKappa, w_acc, cm = calc_cm_rcall(XYTestNFeat[:,-1], yRes)


#     # update the test set by selecting positive samples after prediction 
#     XYTestNFeat_1 = updateTestSetbySelPosSamples(XYTestNFeat, yRes)


#     yRes, clfThres = loadStrongClfDefThres(XYTestNFeat_1, T, './model_stage1/')
#     prec_pos, D1, f1_pos, TPR, F1, Specificity, MCC, CKappa, w_acc, cm = calc_cm_rcall(XYTestNFeat_1[:,-1], yRes)

#     print('Threshold: ', clfThres)
#     F = F0*F1
#     print('F0:',F0)
#     print('F1:',F1)
#     print('F:',F)
#     print('\n')

#     D = D0*D1
#     print('D0:',D0)
#     print('D1:',D1)
#     print('D:',D)
#     print('\n')


#     if D1 > d and F1 < f:
#         clfThresList.append(iThre)
#         F1= F
#         FList.append(F)
#         print('Succeed!')

#     else:

#         iThre = int(clfThres)

#         while iThre > 1 and D1 < d:
#             iThre = iThre - 5

#             yRes = loadStrongClfAdjThres(XYTestNFeat_1, T, './model_stage1/', iThre)
#             prec_pos, D1, f1_pos, TPR, F1, Specificity, MCC, CKappa, w_acc, cm = calc_cm_rcall(XYTestNFeat_1[:,-1], yRes)


#             print('Threshold: ', iThre)
#             F = F0*F1
#             print('F0:',F0)
#             print('F1:',F1)
#             print('F:',F)
#             print('\n')


#             D = D0*D1
#             print('D0:',D0)
#             print('D1:',D1)
#             print('D:',D)
#             print('\n')


#         if D1 > d:
#             if F1 < f:
#                 clfThresList.append(iThre)
#                 F1= F
#                 FList.append(F)
#                 print('Succeed!')
#             else:
#                 print('Fail: F cannot be less than f when D is greater than or equal to d')
#         else:
#             print('Fail, D cannot be greater than or equal to d')



# ## save Positive output as the train set for next stage:

# In[2]:

# print('overall FPRs in each step list: ', FList)
# yLabel = XYTestNFeat[:,-1]
# indList = []
# for i in range(len(yRes)):
#     if yRes[i] == 1 and yLabel[i]==0:
#         indList.append(i)
# print(indList)
# N = XYTest[indList,:]
# N_stage0 = N

# P = XYPosTrn
# XYTrn = np.vstack((P,N))


# ### When # of features = 5, threshold = 10, 
# F0: 0.208053691275
# F1: 0.645161290323
# F: 0.134228187919
# 
# 
# D0: 0.972222222222
# D1: 0.987755102041
# D: 0.960317460317
# 
# ## So stage 2:  features = 5, threshold = 10
# 
# 
# ## Train Stage 3:
# 
# ### Try # of features: 10

# In[ ]:




# In[3]:

# nFeats = 10
# feats = featList[:nFeats]
# #    generate new train set and keep old test set, for the 3rd stage
# XYTrnNFeat = np.hstack((XYTrn[:,feats], XYTrn[:,-1].reshape([-1,1])))
# XYTestNFeat = np.hstack((XYTest[:,feats], XYTest[:,-1].reshape([-1,1])))


# T = 100

# # stage 3
# i = 2
# mdlpath = './model_stage'+str(i)+'/'
# saveFinish = buildStrongClfDefThres(XYTrnNFeat, T, mdlpath)
# assert saveFinish


# #       evaluate cascaded classifier on validation set to determine F and D
# # load stage 1 to test on test set

# XYTestNFeat0 = np.hstack((XYTestNFeat[:,:1].reshape(-1,1), XYTestNFeat[:,-1].reshape([-1,1])))
# print(XYTestNFeat0.shape)

# # load the tuned threshold
# yRes = loadStrongClfAdjThres(XYTestNFeat, T, './model_stage0/',clfThresList[0])
# prec_pos, D0, f1_pos, TPR, F0, Specificity, MCC, CKappa, w_acc, cm = calc_cm_rcall(XYTestNFeat[:,-1], yRes)

# yLabel = XYTestNFeat[:,-1]
# indList = []
# for i in range(len(yRes)):
#     if yRes[i] == 1:
#         indList.append(i)
        
# print(indList)
# XYTestNFeat_1 = XYTestNFeat[indList,:]

# yRes, clfThres = loadStrongClfDefThres(XYTestNFeat_1, T, './model_stage1/')
# prec_pos, D1, f1_pos, TPR, F1, Specificity, MCC, CKappa, w_acc, cm = calc_cm_rcall(XYTestNFeat_1[:,-1], yRes)

# print('Threshold: ', clfThres)
# F = F0*F1
# print('F0:',F0)
# print('F1:',F1)
# print('F:',F)
# print('\n')

# D = D0*D1
# print('D0:',D0)
# print('D1:',D1)
# print('D:',D)
# print('\n')


# iThre = int(clfThres)


# while iThre > 1 and D1 < d:
#     iThre = iThre - 5

#     yRes = loadStrongClfAdjThres(XYTestNFeat_1, T, './model_stage1/', iThre)
#     prec_pos, D1, f1_pos, TPR, F1, Specificity, MCC, CKappa, w_acc, cm = calc_cm_rcall(XYTestNFeat_1[:,-1], yRes)


#     print('Threshold: ', iThre)
#     F = F0*F1
#     print('F0:',F0)
#     print('F1:',F1)
#     print('F:',F)
#     print('\n')


#     D = D0*D1
#     print('D0:',D0)
#     print('D1:',D1)
#     print('D:',D)
#     print('\n')


# if D1 > d:
#     if F1 < f:
#         clfThresList.append(iThre)
#         F1= F
#         FList.append(F)
#     else:
#         print('Fail: F cannot be less than f when D is greater than or equal to d')
# else:
#     print('Fail, D cannot be greater than or equal to d')
    


# In[4]:

# print(D)
# print(f*FList[0])


# In[ ]:

# Life is a journey. Don't get lost. 
# The TV show exibits all kinds of emotional issues. Almost everything in it exceeds the boundary that you can ever imagine. It shows all 
# I Never 
# 
# 

