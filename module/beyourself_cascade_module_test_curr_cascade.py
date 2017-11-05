import os,re
import numpy as np
import pandas as pd
import random
import matplotlib
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, timedelta, time, date
from PASDAC.data import tt_split
from PASDAC.ml import calc_cm_rcall
from PASDAC.util import list_files_in_directory, create_folder
from cascade_func import build_strongclf_def_thres, load_strongclf_def_thres, load_strongclf_adj_thres,\
                            update_XY_with_N_feats, test_cascade_all_stages_keep_true_samples,\
                            update_trnset_with_FP_true_samples, test_cascade_all_stages_real_run,\
                            update_trnset_with_P_samples, read_haar_feat_random_select_samples



class Stage(object):

    def __init__(self, f, d, T, beta_list, model, path_list, count):
        self.f = f
        self.d = d
        self.T = T
        self.beta_list = beta_list
        self.path_list = path_list
        self.model = model
        self.count=count        


    def forward(self, F, D, T_list, all_feat_list, n_feats, n_feats_list, beta_list, thres_list, path_list, XYTrn, XYTest):
        FPrev = F
        DPrev = D
        print('FPrev: ', FPrev)
        print('DPrev: ', DPrev)
        print('\n============== STAGE: ', self.count,' ==============')

        while F > FPrev*self.f and n_feats < len(all_feat_list):
            
            n_feats = n_feats + 1
            print('number of feats:', n_feats)
            
            ######################################################################################
            #       build strong classifier with only first n_feats features in train set 
            ######################################################################################
            
            mdlpath = os.path.join(self.model, 'stage_'+str(self.count))
            create_folder(mdlpath)
        
            XYTrn_nfeats = update_XY_with_N_feats(XYTrn, all_feat_list, n_feats)
            XYTest_nfeats = update_XY_with_N_feats(XYTest, all_feat_list, n_feats)

    #       build stage with features selected
            betas, _ = build_strongclf_def_thres(XYTrn_nfeats, self.T, mdlpath)

    #       evaluate cascaded classifier on validation set to determine F and D
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

            print("  overall recall_pos: ",D)
            print("  overall false positive rate: ", F)
            
            if D > DPrev*self.d and F < FPrev*self.f:
                thres_list.append(clf_thres)
                print('Succeed!')

            elif D < DPrev*self.d:
                while clf_thres > 2 and D < DPrev*self.d:
                    clf_thres = clf_thres - 2
                    # ytrain_res = load_strongclf_adj_thres(XYTrn_nfeats, self.T, clf_thres, mdlpath)

                    if self.count == 0:
                        # y_res, clf_thres = load_strongclf_def_thres(XYTest, self.T, mdlpath)
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
                    # print("  recall_pos: ",D)
                    # print("  false positive rate: ", F)
                    
                if D > DPrev*self.d:
                    if F < FPrev*self.f:
                        thres_list.append(clf_thres)
                        print("  overall recall_pos: ",D)
                        print("  overall false positive rate: ", F)
                        print('Succeed!', '\n')
                    else:
                        print('  Fail: F cannot be less than f when D is greater than or equal to d', '\n')
                else:
                    print('  Fail, D cannot be greater than or equal to d', '\n')

            elif F > FPrev*self.f:
                    print('  Specific case pursued came out: F > f !!! need to tune F, D, f, d', '\n')

        ytrain_res = load_strongclf_adj_thres(XYTrn, self.T, clf_thres, mdlpath)
        XYTrn = update_trnset_with_FP_true_samples(XYTrn, ytrain_res)

        T_list.append(self.T)
        n_feats_list.append(n_feats)
        beta_list.append(betas)
        path_list.append(mdlpath)
        
        self.beta_list=beta_list
        self.path_list=path_list

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

        XYPosTrn, XYPosTest = tt_split(XYPos, self.split)
        XYNegTrn, XYNegTest = tt_split(XYNeg, self.split)

        P = XYPosTrn
        N = XYNegTrn

        XYTrn = np.vstack((P, N))
        print('XYTrn shape:', XYTrn.shape)
        XYTest = np.vstack((XYPosTest,XYNegTest))
        print('XYTest shape:', XYTest.shape)


        self.cascade_stages=[]
        count=0
        for f,d, T in self.parameter:
            stage=Stage(f, d, T, [], self.model, [], count)
            count+=1
            self.cascade_stages.append(stage)

        # get all_feat_list
        _, all_feat_list = build_strongclf_def_thres(XY, T = 10, mdlpath = './tmp_mdl/')
        
        n_feats=0
        F = 1
        D = 1
        n_feats_list = []
        beta_list = []
        thres_list = []
        path_list = []
        T_list = []

        for s in self.cascade_stages:
            F, D, T_list, n_feats, n_feats_list, beta_list, thres_list, path_list, XYTrn = s.forward(F, D, T_list, all_feat_list, n_feats, n_feats_list, beta_list, thres_list, path_list, XYTrn, XYTest)   

        return T_list, all_feat_list, n_feats_list, beta_list, thres_list, path_list


    def test(self, XY, T_list, all_feat_list, n_feats_list, beta_list, thres_list, path_list):
        F_list, D_list, F_final, D_final = test_cascade_all_stages_real_run(XY, T_list, all_feat_list, n_feats_list, beta_list, thres_list, path_list)
        print('Recall(postive) for every stage: ', D_list)
        print('False positive rate for every stage: ', F_list)
        print('Overall recall(postive): ', D_final)
        print('Overall false positive rate: ', F_final)

        return F_list, D_list, F_final, D_final




if __name__ == "__main__":

    feat_folder = '/Volumes/SHIBO/BeYourself/BeYourself/PROCESS/P120/wrist/haar_feature/'
    meals = ['m0824_1','m0824_2']#

    REC = 12
    WINSIZE = 2

    XY = read_haar_feat_random_select_samples(meals, REC, WINSIZE, ratio = 5, use_seed = 1)

    model = Cascaded(
        stage_parameter=[(0.3, 0.7, 100), (0.3, 0.7, 100),(0.3, 0.7, 100)],
        split=0.7,
        model_path="./modularized_cascade_model/"
    )

    T_list, all_feat_list, n_feats_list, beta_list, thres_list, path_list = model.fit(XY)

    XYPos =  XY[np.where(XY[:,-1]==1)[0],:]
    XYNeg =  XY[np.where(XY[:,-1]==0)[0],:]
    print('XYPos:',XYPos.shape)

    XYPosTrn, XYPosTest = tt_split(XYPos, 0.7)
    XYNegTrn, XYNegTest = tt_split(XYNeg, 0.7)

    P = XYPosTrn
    N = XYNegTrn

    XYTrn = np.vstack((P, N))
    print('XYTrn shape:', XYTrn.shape)
    XYTest = np.vstack((XYPosTest,XYNegTest))
    print('XYTest shape:', XYTest.shape)
    print('thres_list: ', thres_list)

    model.test(XYTest, T_list, all_feat_list, n_feats_list, beta_list, thres_list, path_list)



