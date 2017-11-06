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
from cascade_strong import build_strongclf_def_thres, load_strongclf_def_thres, load_strongclf_adj_thres,\
                            update_XY_with_N_feats,\
                            update_trnset_with_FP_true_samples

from beyourself_cascade import read_haar_feat_random_select_samples



class Stage(object):
    """docstring for Stage"""
    def __init__(self, f, d, T, beta_list, model, path_list, count):
        self.f = f
        self.d = d
        self.T = T
        self.beta_list = beta_list
        self.path_list = path_list
        self.model = model
        self.count=count        


    def forward(self, F, all_feat_list, n_feats, n_feats_list, beta_list, thres_list, path_list, XYTrn, XYTest):
        FPrev = F
        print('FPrev: ', FPrev)
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

    #       build stage with features selected
            betas, _ = build_strongclf_def_thres(XYTrn_nfeats, self.T, mdlpath)

            # !!!!!! SHOULD LOAD ALL THE PREVIOUS STAGES, READ FROM PATH_LIST
    #       evaluate cascaded classifier on validation set to determine F and D
            if self.count == 0:
                yRes, clfThres = load_strongclf_def_thres(XYTest, self.T, mdlpath)
            else:
                # SHOULD READ ALL PREVIOUS THRES !!!!!!! THIS IS ONLY FOR TWO STAGE TEST
                yRes = load_strongclf_adj_thres(XYTest, self.T, thres_list[0], mdlpath)
                clfThres = thres_list[0]


            print("  clf thres: ", clfThres)
            prec_pos, D, f1_pos, TPR, F, Specificity, MCC, CKappa, w_acc, cm = calc_cm_rcall(XYTest[:,-1], yRes)
            print("  recall_pos: ",D)
            print("  false positive rate: ", F)
            
            if D > self.d and F < FPrev*self.f:
                thres_list.append(clfThres)
                print('Succeed!')
                FList.append(F)

            elif D < self.d:
                thres = int(clfThres)
                while thres > 2 and D < self.d:
                    thres = thres - 2
                    yRes = load_strongclf_adj_thres(XYTrn_nfeats, self.T, thres, mdlpath)
                    print("  clf thres: ", thres)
                    # WHY USE XYTrn_nfeats INSTEAD OF XYTEST@@!!!!!!
                    prec_pos, D, f1_pos, TPR, F, Specificity, MCC, CKappa, w_acc, cm = calc_cm_rcall(XYTrn_nfeats[:,-1], yRes)
                    print("  recall_pos: ",D)
                    print("  false positive rate: ", F)
                    
                if D > self.d:
                    if F < FPrev*self.f:
                        thres_list.append(thres)
                        print('Succeed!', '\n')
                    else:
                        print('  Fail: F cannot be less than f when D is greater than or equal to d', '\n')
                else:
                    print('  Fail, D cannot be greater than or equal to d', '\n')

            elif F > FPrev*self.f:
                    print('  Specific case pursued came out: F > f !!! need to tune F, D, f, d', '\n')

        XYTrn = update_trnset_with_FP_true_samples(XYTrn, yRes)

        n_feats_list.append(n_feats)
        beta_list.append(betas)
        path_list.append(mdlpath)
        
        self.beta_list=beta_list
        self.path_list=path_list

        return F, n_feats, n_feats_list, beta_list, thres_list, path_list, XYTrn
        

    # def test(self, XY, T_list, all_feat_list, n_feats_list, beta_list, thres_list, path_list):
    #     return test_cascade_all_stages(XY, T_list, all_feat_list, n_feats_list, beta_list, thres_list, path_list)
        



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

        # P N: USE P AND N !!!!!!
        P = XYPosTrn
        N = XYNegTrn

        XYTrn = np.vstack((P, N))
        print('XYTrn shape:', XYTrn.shape)
        XYTest = np.vstack((XYPosTest,XYNegTest))
        print('XYTest shape:', XYTest.shape)


        self.cascade_stages=[]
        F = 1
        count=0
        for f,d, T in self.parameter:
            stage=Stage(f, d, T, [], self.model, [], count)
            count+=1
            self.cascade_stages.append(stage)

        _, all_feat_list = build_strongclf_def_thres(XY, T = 10, mdlpath = './tmp_mdl/')
        
        n_feats=0
        D = 1
        n_feats_list = []
        beta_list = []
        thres_list = []
        path_list = []

        for s in self.cascade_stages:
            _,n_feats, n_feats_list, beta_list, thres_list, path_list, XYTrn = s.forward(F, all_feat_list, n_feats, n_feats_list, beta_list, thres_list, path_list, XYTrn, XYTest)   

        print(n_feats_list) 




if __name__ == "__main__":

    # !!!!!!!! 
    # FUNCTION updateTrnUsingFP_allTrueSamples IS NOT SHOWN IN THIS VERSION

    feat_folder = '/Volumes/SHIBO/BeYourself/BeYourself/PROCESS/P120/wrist/haar_feature/'
    meals = ['m0824_1','m0824_2']#

    REC = 12
    WINSIZE = 2

    XY = read_haar_feat_random_select_samples(meals, REC, WINSIZE, ratio = 5, use_seed = 1)

    model = Cascaded(
        stage_parameter=[(0.3, 0.7, 100), (0.3, 0.7, 100)],
        split=0.7,
        model_path="./modularized_cascade_model/"
    )

    model.fit(XY)






