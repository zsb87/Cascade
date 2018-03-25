import os
import re
import random
import numpy as np
from sklearn import preprocessing
from cascade_util import __pred_weakclf, __weval, __w_beta_update, __save_weakclf, __load__weakclf
import operator
import functools
import pandas as pd
from sklearn.metrics import *
from sklearn.metrics import cohen_kappa_score


def calc_cm_rcall(y_test, y_pred):#

    ct = pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)

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

    recall_pos = TP/(TP+FN)

    # F1 score for positive = 2 * precision * recall / (precision + recall)….or it can be F1= 2*TP/(2*TP + FP+ FN)
    f1_pos = 2*TP/(TP*2 + FP+ FN)
    # TPR = TP/(TP+FN)
    TPR = cm[1,1]/sum(cm[1,j] for j in range(len(set(y_test))))
    # FPR = FP/(FP+TN)
    FPR = cm[0,1]/sum(cm[0,j] for j in range(len(set(y_test))))
    # specificity = TN/(FP+TN)
    Specificity = cm[0,0]/sum(cm[0,j] for j in range(len(set(y_test))))
    MCC = matthews_corrcoef(y_test, y_pred)
    CKappa = cohen_kappa_score(y_test, y_pred)

    # w_acc = (TP*20 + TN)/ [(TP+FN)*20 + (TN+FP)] if 20:1 ratio of non-feeding to feeding
    ratio = (TN+FP)/(TP+FN)
    w_acc = (TP*ratio + TN)/ ((TP+FN)*ratio + (TN+FP))

    return prec_pos, recall_pos, f1_pos, TPR, FPR, Specificity, MCC, CKappa, w_acc, cm


def rm_files_in_folder(folder):
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


def list_files_in_directory(mypath):
    
    return [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]



# def read_raw_acc_separate_files(meal, rec, winsize):
#     folder = os.path.join('/Volumes/SHIBO/BeYourself/BeYourself/PROCESS/P120/wrist/haar_feature/', meal)
#     allfiles = list_files_in_directory(folder)
#     file_header = 'feat_rec'+str(rec)+'_label_win'+str(winsize)+'_'
#     RegExr= file_header+'\d+.txt'
#     matches = [re.search(RegExr, f) for f in allfiles]
#     files = ([m.group() for m in matches if m])

#     # files = [files[0], files[1]]
#     XY = [np.loadtxt(os.path.join(folder, file), delimiter=",", unpack=False) for file in files]
#     XY = np.vstack(XY)
#     return XY



def read_haar_feat_raw_separate_files(meal, rec, winsize):
    folder = os.path.join('/Volumes/SHIBO/BeYourself/BeYourself/PROCESS/P120/wrist/haar_feature/', meal)
    allfiles = list_files_in_directory(folder)
    file_header = 'feat_rec'+str(rec)+'_label_win'+str(winsize)+'_'
    RegExr= file_header+'\d+.txt'
    matches = [re.search(RegExr, f) for f in allfiles]
    files = ([m.group() for m in matches if m])

    # files = [files[0], files[1]]
    XY = [np.loadtxt(os.path.join(folder, file), delimiter=",", unpack=False) for file in files]
    XY = np.vstack(XY)
    return XY



def read_haar_feat_random_select_samples(meal, rec, winsize, ratio, use_seed):
    # ratio: negative to positive samples

    if isinstance(meal,str):
        folder = os.path.join('/Volumes/SHIBO/BeYourself/BeYourself/PROCESS/P120/wrist/haar_feature/', meal)
        pos_file = 'feat_rec'+str(rec)+'_label_win'+str(winsize)+'_pos.txt'
        neg_file = 'feat_rec'+str(rec)+'_label_win'+str(winsize)+'_neg.txt'
        XYPos = np.loadtxt(os.path.join(folder, pos_file), delimiter=",", unpack=False)
        XYNeg = np.loadtxt(os.path.join(folder, neg_file), delimiter=",", unpack=False)
        if use_seed: random.seed(1)
        rand_ind = random.sample(range(0, XYNeg.shape[0]), XYPos.shape[0]*ratio)
        XYNeg = XYNeg[rand_ind, :]

        XY = np.vstack((XYPos, XYNeg))

    elif isinstance(meal,list):
        meals = meal
        XYs = []
        for meal in meals:
            folder = os.path.join('/Volumes/SHIBO/BeYourself/BeYourself/PROCESS/P120/wrist/haar_feature/', meal)
            pos_file = 'feat_rec'+str(rec)+'_label_win'+str(winsize)+'_pos.txt'
            neg_file = 'feat_rec'+str(rec)+'_label_win'+str(winsize)+'_neg.txt'
            XYPos = np.loadtxt(os.path.join(folder, pos_file), delimiter=",", unpack=False)
            XYNeg = np.loadtxt(os.path.join(folder, neg_file), delimiter=",", unpack=False)
            if use_seed: random.seed(1)
            rand_ind = random.sample(range(0, XYNeg.shape[0]), XYPos.shape[0]*ratio)
            XYNeg = XYNeg[rand_ind, :]

            XY = np.vstack((XYPos, XYNeg))
            XYs.append(XY)
        XY = np.vstack(XYs)

    return XY



def build_strongclf_def_thres(XY, T, mdlpath, verbose = 0):
    """
    Parameters
    ---------- 
    XY:   features and labels
    T:   number of iterations, IMPORTANT PARAMETER
    mdlpath:   the path where the models are saved

    Save
    ----
    model
    f_opt_ind: feature index
    beta
    """

    # re-order negative and positive samples to match the weight vector

    first_time_flag = 1

    XYNeg =  XY[np.where(XY[:,-1]==0)[0],:]
    XYPos =  XY[np.where(XY[:,-1]==1)[0],:]
    XY = np.vstack((XYNeg, XYPos))

    y_label = XY[:,-1]

    nsamples = XYNeg.shape[0]
    psamples = XYPos.shape[0]
    samples = nsamples + psamples
    
    # number of features
    n_feats = XY.shape[1]-1
    f_list = list(range(n_feats))

    if verbose: print('f_list: ', f_list)

    if not os.path.exists(mdlpath):
        os.makedirs(mdlpath)
    else:
        rm_files_in_folder(mdlpath)
        
    f_opt_list = []
    
    betas = np.zeros([T])# keep record of all betas in all rounds

    if verbose: w_norm_rec = np.zeros([T, samples])

    # initialize weights
    w = np.zeros([samples])
    w[:nsamples] = 1/nsamples
    w[nsamples:] = 1/psamples


    # Iteration
    for t in range(T):

        # 1. normalize weights
        w_norm = preprocessing.normalize(w[:,np.newaxis], axis=0, norm = 'l1')

        if verbose: print(w_norm)
        if verbose: w_norm_rec[t,:] = w_norm.reshape((1,-1)) # w_norm record

        err = np.ones(n_feats)

        # for each feature:
        for f in f_list:

            # 2. train a classifier h using a single feature.
            w_norm = w_norm.ravel()        
            y_pred = __pred_weakclf(XY[:,f].reshape(-1, 1), y_label, w_norm)

            # error calculation
            err[f] = __weval(y_label, y_pred, w_norm)

        if first_time_flag:
            first_time_flag = 0
            if verbose > 1: 
                print('err: ', err)
                print(sorted(range(len(err)), key=lambda k: err[k]))

        # 3. choose the classifier with lowest error
        f_opt_ind = np.argmin(err)
        if verbose and t%(T/10): print('f_opt_ind: ',f_opt_ind) 

        errmin = np.amin(err)        
        if verbose and t%(T/10) == 0: print('errmin: ',errmin) 

        f_opt_list.append(f_opt_ind)

        # 4. update weights
        w, betas = __w_beta_update(betas, t, errmin, w_norm, samples, y_label, y_pred)

        __save_weakclf(XY[:,np.argmin(err)].reshape(-1, 1), y_label, \
                       os.path.join(mdlpath,str(t)+'.sav'), \
                       f_opt_ind, betas[t],\
                       os.path.join(mdlpath,str(t)+'_conf.txt'), \
                       w_norm)

        feat_list = sorted(range(len(err)), key=lambda k: err[k])

    if verbose: 
        print('f_opt_list: ', f_opt_list) 
        print('err: ', err)
        print(feat_list)

    return betas, feat_list



def load_strongclf_def_thres(XY, T, mdlpath, verbose = 0):
    """
    Load weak classifier and build final strong classifier with default threshold 
    after training with training set.
    threshold is not adjustable.

    the models are './model/(i).sav'

    Parameters
    ---------- 
    XY:   features and labels
    T:   number of iterations, IMPORTANT PARAMETER
    mdlpath:   the path where the models are saved

    """

    
#   load and get weak classifiers result
    y_pred_rec = np.zeros([T, XY.shape[0]])
    y_res = np.zeros(XY.shape[0])
    X = XY[:,:-1]
    betas = np.zeros([T])# keep record of all betas in all rounds

    for t in range(T):
        h, feat, betas[t] = __load__weakclf(os.path.join(mdlpath,str(t)+'.sav'), \
                                            os.path.join(mdlpath,str(t)+'_conf.txt'))

        if verbose:
            print('feat: ',feat)
            print('X[:,feat]:', X[:,feat])
        y_pred = h.predict(X[:,int(feat)].reshape(-1, 1))
        y_pred_rec[t,:] = y_pred
        
#     calc classifier threshold
    betas_recip = np.reciprocal(betas)
    alphas = np.log(betas_recip)
    clf_thres = np.sum(alphas)*0.5
    
    y_comb = np.dot(alphas, y_pred_rec)
    
#     get final result

    for i in range(y_comb.shape[0]):
        if y_comb[i] < clf_thres:
            y_res[i] = 0
        else:
            y_res[i] = 1
    
    return y_res, clf_thres



def load_strongclf_adj_thres(XY, T, clf_thres, mdlpath):
    """
    Final strong classifier with adjustable threshold

    Load weak classifier and build final strong classifier with adjustable threshold 
    after training with training set.

    the models are './model/(i).sav'

    Parameters
    ---------- 
    XY:   features and labels
    T:   number of iterations, IMPORTANT PARAMETER
    mdlpath:   the path where the models are saved

    """



#     load and get weak classifiers result
    y_predRec = np.zeros([T, XY.shape[0]])
    y_res = np.zeros(XY.shape[0])
    X = XY[:,:-1]
    betas = np.zeros([T])# keep record of all betas in all rounds

    for t in range(T):
        h, feat, betas[t] = __load__weakclf(os.path.join(mdlpath,str(t)+'.sav'),\
                                            os.path.join(mdlpath,str(t)+'_conf.txt'))
        yPred = h.predict(X[:,feat].reshape(-1, 1))
        y_predRec[t,:] = yPred
        
#     calc classify threshold
    betas_recip = np.reciprocal(betas)
    alphas = np.log(betas_recip)
    
    y_comb = np.dot(alphas, y_predRec) 
    
#     print(y_comb)

    for i in range(y_comb.shape[0]):
        if y_comb[i] < clf_thres:
            y_res[i] = 0
        else:
            y_res[i] = 1
    
    return y_res

def update_trnset_with_P_samples(XY, y_res):
    # ind_list = []
    # for i in range(len(y_res)):
    #     if y_res[i] == 1:
    #         ind_list.append(i)
    # P = XY[ind_list,:]
    P = XY[np.where(y_res==1)[0],:]

    return P


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
    


def update_trnset_with_FP_true_samples(XYCurrTrn, yRes):
    yLabel = XYCurrTrn[:,-1]
    indList = []
    
    for i in range(len(yRes)):
        if yRes[i] == 1 and yLabel[i]==0:
            indList.append(i)
            
    N = XYCurrTrn[indList,:]
    P = XYCurrTrn[np.where(XYCurrTrn[:,-1]==1)[0],:]

    XYTrn = np.vstack((P,N))

    return XYTrn
    


def update_index_trnset_with_FP_true_samples(XYCurrTrn, yRes):
    yLabel = XYCurrTrn[:,-1]
    FP_ind_list = []
    
    for i in range(len(yRes)):
        if yRes[i] == 1 and yLabel[i]==0:
            FP_ind_list.append(i)
            
    index = FP_ind_list + np.where(XYCurrTrn[:,-1]==1)

    return index
    


def update_XY_with_FP_P_samples(XYPosTrn, yRes, XYTestNFeat):
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


def update_XY_with_N_feats(XYTrn, featList, nFeats):
#   select first n features in featList for training set
    feats = featList[:nFeats]
    XYTrn_nfeats = np.hstack((XYTrn[:,feats], XYTrn[:,-1].reshape([-1,1])))

    return XYTrn_nfeats


# SHOULE BE REMOVED, REPLACED BY  update_XY_with_N_feats
def XYTrnUpdateWithTopNFeats(XYTrn, featList, nFeats):
#   select first n features in featList for training set
    feats = featList[:nFeats]
    XYTrnNFeat = np.hstack((XYTrn[:,feats], XYTrn[:,-1].reshape([-1,1])))

    return XYTrnNFeat



def test_cascade_all_stages_keep_true_samples(XY, T_list, all_feat_list, n_feats_list, beta_list, thres_list, path_list):
    """
    GT Positive samples P keep going through the full stages
    True negative samples are dropped out
    """

    # initialization
    stages = len(T_list)
    F_list = []
    D_list = []

    # for-loop
    for stage in range(stages):
        # select XY with selected features to update XY
        XY_stage = update_XY_with_N_feats(XY, all_feat_list, n_feats_list[stage])
        y_res = load_strongclf_adj_thres(XY_stage, T_list[stage], thres_list[stage], path_list[stage])
        _, D, _, _, F, _, _, _, _, _ = calc_cm_rcall(XY[:,-1], y_res)

    # update dataset
        XY = update_trnset_with_FP_true_samples(XY, y_res)
        
        F_list.append(F)
        D_list.append(D)

    F_final = functools.reduce(operator.mul, F_list, 1)
    D_final = functools.reduce(operator.mul, D_list, 1)

    return F_list, D_list, F_final, D_final, XY




def test_cascade_all_stages_real_run(XY, T_list, all_feat_list, n_feats_list, beta_list, thres_list, path_list):
    """
    All positive samples go into the next stage,
    all negative samples are dropped out
    """

    # initialization
    stages = len(T_list)
    F_list = []
    D_list = []

    # for-loop
    for stage in range(stages):
        # select XY with selected features to update XY
        XY_stage = update_XY_with_N_feats(XY, all_feat_list, n_feats_list[stage])
        y_res = load_strongclf_adj_thres(XY_stage, T_list[stage], thres_list[stage], path_list[stage])
        _, D, _, _, F, _, _, _, _, _ = calc_cm_rcall(XY[:,-1], y_res)

    # update dataset
        XY = update_trnset_with_P_samples(XY, y_res)

        F_list.append(F)
        D_list.append(D)

    F_final = functools.reduce(operator.mul, F_list, 1)
    D_final = functools.reduce(operator.mul, D_list, 1)

    return F_list, D_list, F_final, D_final
