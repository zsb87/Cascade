# 
# Question: Algorithm:1. in Adaboost framework, in for loop, step2: "2. For each feature, train a classifier which is restricted to using a single feature. 
#                       The error is evaluated with respect to w". Question is when I train the weak classifier, 
#                       should I take the weights into my training process? 
# 
#                       -- In my implementation, the weight is used in training
# 
#                     2.[IMPORTANT] in Cascade framework, 'Decrease threshold for the i-th classifier until the current cascaded classifier has a detection rate of at least d*D_{i-1}
#                       when decreasing the threshold, what is the termination condition.
# 
# 
# 
#                     3. what if the base classfiers have different direction, for example one has >threshold then 1, another <threshold then 1?
#                        does that matter?
#                           Answer: as the g(x) has wrapped the direction in it, the > symbol in the final decision function(linear combination) is not the judge in base classifier
#                                   so no need to worry
# 
#                     4. what if error rate for one base classifier is <0.5
#                           Answer: then alpha<0, the alpha in linear combination function will recitify/flip it automatically.
# 
# 
# todo: 
#  q: why threshold >0
# 
# 


import os,re
import numpy as np
import pandas as pd
import random
import matplotlib
import matplotlib.pyplot as plt
import warnings
from cascade_func import build_strongclf_def_thres, load_strongclf_def_thres, load_strongclf_adj_thres,\
                            update_XY_with_N_feats, test_cascade_all_stages_keep_true_samples,\
                            update_trnset_with_FP_true_samples, test_cascade_all_stages_real_run,\
                            update_trnset_with_P_samples
from dataset_import import loadDataset, loadDataset_beyourself
from sklearn.metrics import *
from util import create_folder, lprint, tt_split, calc_cm_rcall






class Stage(object):
    """
    Parameters
    ----------     
    f: false positive rate
    d: postive recall (detection rate)
    T: number of iterations
    beta_list: 
    model: the path where models are saved
    path_list:
    count: attribute, route control function

    """

    def __init__(self, f, d, T, beta_list, model, path_list, count, logfile):
        self.f = f
        self.d = d
        self.T = T
        self.beta_list = beta_list
        self.model = model
        self.path_list = path_list
        self.count = count        
        self.logfile = logfile


    def forward(self, F, D, n_feats_max, T_list, all_feat_list, n_feats, n_feats_list, beta_list, thres_list, path_list, XYTrn, XYTest):
        FPrev = F
        DPrev = D
        lprint(self.logfile, '  FPrev: ', FPrev)
        lprint(self.logfile, '  DPrev: ', DPrev)
        lprint(self.logfile, '\n============== STAGE: ', self.count,' ==============')

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
                print( 'DEBUG: betas: ', betas)

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
    """
    Parameters
    ----------     
    stage_parameter: a list, of which each element contains (f, d, T)

        f: false positive rate
        d: postive recall (detection rate)
        T: number of iterations

    split: train set to validation set ratio

    model_path: the path where models are saved

    """


    def __init__(self, stage_parameter, split, model_path, n_feats_max, logfile):
        self.parameter = stage_parameter
        self.split = split
        self.model = model_path
        self.n_feats_max = n_feats_max
        self.logfile = logfile
        create_folder(self.model)


    def fit(self, XY):
        """
        split XY into train set and validation set
        """
        XYPos =  XY[np.where(XY[:,-1]==1)[0],:]
        XYNeg =  XY[np.where(XY[:,-1]==0)[0],:]
        lprint(self.logfile, 'XYPos:', XYPos.shape)
        lprint(self.logfile, 'XYNeg:', XYNeg.shape)

        XYPosTrn, XYPosVal = tt_split(XYPos, self.split)
        XYNegTrn, XYNegVal = tt_split(XYNeg, self.split)

        XYTrn = np.vstack((XYPosTrn, XYNegTrn))
        lprint(self.logfile, 'XYTrn shape:', XYTrn.shape, ', Pos:', XYPosTrn.shape, ', Neg:', XYNegTrn.shape)
        XYVal = np.vstack((XYPosVal,XYNegVal))
        lprint(self.logfile, 'XYVal shape:', XYVal.shape, ', Pos:', XYPosVal.shape, ', Neg:', XYNegVal.shape)


        """
        build cascade
        """
        self.cascade_stages = []
        count = 0

        for f, d, T in self.parameter:
            stage = Stage(f, d, T, [], self.model, [], count, logfile=self.logfile)
            count += 1 
            self.cascade_stages.append(stage)

        # get all_feat_list
        _, all_feat_list = build_strongclf_def_thres(XY, T=10, mdlpath='./tmp_mdl/')
        
        n_feats=0
        F = 1
        D = 1
        n_feats_max = self.n_feats_max
        n_feats_list = []
        beta_list = []
        thres_list = []
        path_list = []
        T_list = []

        for s in self.cascade_stages:

            F, D, T_list, n_feats, n_feats_list, beta_list, thres_list, path_list, XYTrn = \
                s.forward(F, D, n_feats_max, T_list, all_feat_list, n_feats, n_feats_list, beta_list, thres_list, path_list, XYTrn, XYVal)   
            
            lprint(self.logfile, 'feat list: ', all_feat_list[:n_feats])

        return T_list, all_feat_list, n_feats_list, beta_list, thres_list, path_list


    def test(self, XY, T_list, all_feat_list, n_feats_list, beta_list, thres_list, path_list):
        F_list, D_list, F_final, D_final, y_res = test_cascade_all_stages_real_run(XY, T_list, all_feat_list, n_feats_list, beta_list, thres_list, path_list)

        return F_list, D_list, F_final, D_final, y_res






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
