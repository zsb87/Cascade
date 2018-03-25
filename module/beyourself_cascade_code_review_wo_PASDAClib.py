import os,re
import numpy as np
import pandas as pd
import random
import matplotlib
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, timedelta, time, date
from cascade_func import build_strongclf_def_thres, load_strongclf_def_thres, load_strongclf_adj_thres,\
                            update_XY_with_N_feats, test_cascade_all_stages_keep_true_samples,\
                            update_trnset_with_FP_true_samples, test_cascade_all_stages_real_run,\
                            update_trnset_with_P_samples, read_haar_feat_random_select_samples
from dataset_import import loadDataset, loadDataset_beyourself
from sklearn.metrics import *
from sklearn.metrics import cohen_kappa_score







def tt_split(XY, train_ratio):
    # eg: train_ratio = 0.7
    length = len(XY)
    # print(length)
    test_enum = range(int((1-train_ratio)*10))
    test_ind = []
    for i in test_enum:
        test_ind = test_ind + list(range(i, length, 10))

    # test_ind = np.arange(n, length, k)
    train_ind = [x for x in list(range(length)) if x not in test_ind]

    return XY[train_ind], XY[test_ind]


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

    
def create_folder(f, deleteExisting=False):
    '''
    Create the folder

    Parameters:
            f: folder path. Could be nested path (so nested folders will be created)

            deleteExising: if True then the existing folder will be deleted.

    '''
    if os.path.exists(f):
        if deleteExisting:
            shutil.rmtree(f)
    else:
        os.makedirs(f)


def list_files_in_directory(mypath):

    return [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]



class Stage(object):

    def __init__(self, f, d, T, beta_list, model, path_list, count):
        self.f = f
        self.d = d
        self.T = T
        self.beta_list = beta_list
        self.path_list = path_list
        self.model = model
        self.count=count        


    def forward(self, F, D, n_feats_max, T_list, all_feat_list, n_feats, n_feats_list, beta_list, thres_list, path_list, XYTrn, XYTest):
        FPrev = F
        DPrev = D
        print('  FPrev: ', FPrev)
        print('  DPrev: ', DPrev)
        print('\n============== STAGE: ', self.count,' ==============')

        while F > FPrev*self.f:
            
            n_feats = n_feats + 1
            print('  number of feats:', n_feats)

            if n_feats > n_feats_max:
                print('Exceed max number of features.')
                break

            else:                
                
                mdlpath = os.path.join(self.model, 'stage_'+str(self.count))
                create_folder(mdlpath)

                # build strong classifier with only first n_feats features in train set 
                XYTrn_nfeats = update_XY_with_N_feats(XYTrn, all_feat_list, n_feats)
                XYTest_nfeats = update_XY_with_N_feats(XYTest, all_feat_list, n_feats)

                # build stage with features selected
                betas, _ = build_strongclf_def_thres(XYTrn_nfeats, self.T, mdlpath)

                # evaluate cascaded classifier on validation set to determine F and D
                if self.count == 0:
                    ytest_res, clf_thres = load_strongclf_def_thres(XYTest_nfeats, self.T, mdlpath)
                    print("  clf thres: ", clf_thres)
                    _, D, _, _, F, _, _, _, _, _ = calc_cm_rcall(XYTest_nfeats[:,-1], ytest_res)

                else:
                    print( 'DEBUG: thres_list: ', thres_list)
                    print( 'DEBUG: T_list: ', T_list)
                    F_res_list, D_res_list, F_prev_all, D_prev_all, XYTest_prev_stage = test_cascade_all_stages_keep_true_samples(XYTest, T_list, all_feat_list, n_feats_list, beta_list, thres_list, path_list)
                    XYTest_prev_stage_nfeats = update_XY_with_N_feats(XYTest_prev_stage, all_feat_list, n_feats)
                    ytest_res, clf_thres = load_strongclf_def_thres(XYTest_prev_stage_nfeats, self.T, mdlpath)
                    print("  clf thres: ", clf_thres)
                    _, D, _, _, F, _, _, _, _, _ = calc_cm_rcall(XYTest_prev_stage_nfeats[:,-1], ytest_res)
                    F = F*F_prev_all
                    D = D*D_prev_all
                    print("  clf thres: ", clf_thres)
                
                while D < DPrev*self.d:

                    if clf_thres > 2:
                        clf_thres = clf_thres - 2

                        if self.count == 0:
                            ytest_res = load_strongclf_adj_thres(XYTest_nfeats, self.T, clf_thres, mdlpath)
                            _, D, _, _, F, _, _, _, _, _ = calc_cm_rcall(XYTest_nfeats[:,-1], ytest_res)

                        else:
                            F_res_list, D_res_list, F_prev_all, D_prev_all, XYTest_prev_stage = test_cascade_all_stages_keep_true_samples(XYTest, T_list, all_feat_list, n_feats_list, beta_list, thres_list, path_list)
                            XYTest_prev_stage_nfeats = update_XY_with_N_feats(XYTest_prev_stage, all_feat_list, n_feats)
                            ytest_res = load_strongclf_adj_thres(XYTest_prev_stage_nfeats, self.T, clf_thres, mdlpath)
                            _, D, _, _, F, _, _, _, _, _ = calc_cm_rcall(XYTest_prev_stage_nfeats[:,-1], ytest_res)
                            F = F*F_prev_all
                            D = D*D_prev_all

                        print("  clf thres: ", clf_thres)

                    else:
                        break

        ytrain_res = load_strongclf_adj_thres(XYTrn, self.T, clf_thres, mdlpath)
        XYTrn = update_trnset_with_FP_true_samples(XYTrn, ytrain_res)

        T_list.append(self.T)
        n_feats_list.append(n_feats)
        beta_list.append(betas)
        thres_list.append(clf_thres)
        path_list.append(mdlpath)
        
        self.beta_list = beta_list
        self.path_list = path_list

        return F, D, T_list, n_feats, n_feats_list, beta_list, thres_list, path_list, XYTrn
        

        

class Cascaded(object):

    def __init__(self, stage_parameter, split, model_path):
        self.parameter = stage_parameter
        self.split = split
        self.model = model_path
        create_folder(self.model)


    def fit(self, XY):
        XYPos =  XY[np.where(XY[:,-1]==1)[0],:]
        XYNeg =  XY[np.where(XY[:,-1]==0)[0],:]
        print('XYPos:',XYPos.shape)

        XYPosTrn, XYPosVal = tt_split(XYPos, self.split)
        XYNegTrn, XYNegVal = tt_split(XYNeg, self.split)

        P = XYPosTrn
        N = XYNegTrn

        XYTrn = np.vstack((P, N))
        print('XYTrn shape:', XYTrn.shape)
        XYVal = np.vstack((XYPosVal,XYNegVal))
        print('XYVal shape:', XYVal.shape)


        self.cascade_stages = []
        count = 0
        for f,d, T in self.parameter:
            stage = Stage(f, d, T, [], self.model, [], count)
            count += 1 
            self.cascade_stages.append(stage)

        # get all_feat_list
        _, all_feat_list = build_strongclf_def_thres(XY, T = 10, mdlpath = './tmp_mdl/')
        
        n_feats=0
        F = 1
        D = 1
        n_feats_max = 600
        n_feats_list = []
        beta_list = []
        thres_list = []
        path_list = []
        T_list = []

        for s in self.cascade_stages:
            F, D, T_list, n_feats, n_feats_list, beta_list, thres_list, path_list, XYTrn = s.forward(F, D, n_feats_max, T_list, all_feat_list, n_feats, n_feats_list, beta_list, thres_list, path_list, XYTrn, XYVal)   

        return T_list, all_feat_list, n_feats_list, beta_list, thres_list, path_list


    def test(self, XY, T_list, all_feat_list, n_feats_list, beta_list, thres_list, path_list):
        F_list, D_list, F_final, D_final = test_cascade_all_stages_real_run(XY, T_list, all_feat_list, n_feats_list, beta_list, thres_list, path_list)
        print('Recall(postive) for every stage: ', D_list)
        print('False positive rate for every stage: ', F_list)
        print('Overall recall(postive): ', D_final)
        print('Overall false positive rate: ', F_final)

        return F_list, D_list, F_final, D_final






if __name__ == "__main__":

    # dataset = 'feeding'
    # XY = loadDataset(dataset)


    dataset = 'beyourself_P120' 
    meal = 'm0824_1'
    XYTest = loadDataset_beyourself(dataset, meal)

    # IMPORT DATA

    subj = 'P120'

    OUT_DIR =  '/Volumes/Seagate/SHIBO/BeYourself-Structured/OUTPUT'
    OUT_FOLDER = os.path.join(OUT_DIR, subj, 'MEAL')



    # feat_folder = os.path.join(OUT_FOLDER, meal, 'FEATURE')

    # names_file_path = os.path.join(feat_folder, 'feat_type12_label_win'+str(FRAME_SIZE_SEC)+'_str'+str(STEP_SIZE_SEC)+'_names.txt')
    # # lines = tuple(open(names_file_path, 'r'))
    # with open(names_file_path, "r") as ins:
    #     names = []
    #     for line in ins:
    #         names.append(line[:-2])



    df = pd.read_csv(os.path.join(OUT_FOLDER,'LOPO/leave_m0824_1/FEATURE','dev_accgyr_type12_label_win2_str1.0_overlapratio0.75_all_len_fixed.txt'))
    XYTrn = df.as_matrix()

    # XYPos =  XY[np.where(XY[:,-1]==1)[0],:]
    # XYNeg =  XY[np.where(XY[:,-1]==0)[0],:]
    # print('XYPos:',XYPos.shape)
    # print('XYNeg:',XYNeg.shape)

    # XYPosTrn, XYPosTest = tt_split(XYPos, 0.7)
    # XYNegTrn, XYNegTest = tt_split(XYNeg, 0.7)

    # XYTrn = np.vstack((XYPosTrn, XYNegTrn))
    # print('XYTrn shape:', XYTrn.shape)
    # XYTest = np.vstack((XYPosTest,XYNegTest))
    # print('XYTest shape:', XYTest.shape)


    # cascaded classifier configuration

    model = Cascaded(
        stage_parameter=[(0.3, 0.8, 100)],
        split=0.7,
        model_path="./modularized_cascade_model/"
    )

    # train/fit

    T_list, all_feat_list, n_feats_list, beta_list, thres_list, path_list = model.fit(XYTrn)

    # test

    model.test(XYTest, T_list, all_feat_list, n_feats_list, beta_list, thres_list, path_list)
