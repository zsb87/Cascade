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
from PASDAC.util import list_files_in_directory
from cascade_strong import build_strongclf_def_thres, load_strongclf_def_thres, load_strongclf_adj_thres,\
                            XYTrnUpdateWithTopNFeats, updateTrnsetWithFPtrueSamples,\
                            update_trnset_with_FP_true_samples


# todo: in test set, when segment has overlap with feeding gesture but overlap score < 7.5 and regarded as non-feeding, 
#       needs to be removed from test set

def read_raw_acc_separate_files(meal, rec, winsize):
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



if __name__ == "__main__":

    feat_folder = '/Volumes/SHIBO/BeYourself/BeYourself/PROCESS/P120/wrist/haar_feature/'
    meals = ['m0824_1','m0824_2']#

    REC = 12
    WINSIZE = 2



    # ##--------------------------------------------------------------------------------------
    # # SAVE POSITIVE SAMPLES IN A SINGLE FILE
    # XY = read_haar_feat_raw_separate_files(meal, REC, WINSIZE)
    # XYPos =  XY[np.where(XY[:,-1]==1)[0],:]
    # XYNeg =  XY[np.where(XY[:,-1]==0)[0],:]
    # print('XYPos:',XYPos.shape)
    # print('XYNeg:',XYNeg.shape)

    # # savename = 'feat_rec'+str(REC)+'_label_win'+str(WINSIZE)+'_pos.txt'
    # savename = 'accXYZ_label_win'+str(WINSIZE)+'_pos.txt'
    # np.savetxt(os.path.join(feat_folder, meal, savename), XYPos, delimiter=",")
    # # savename = 'feat_rec'+str(REC)+'_label_win'+str(WINSIZE)+'_neg.txt'
    # savename = 'accXYZ_label_win'+str(WINSIZE)+'_neg.txt'
    # np.savetxt(os.path.join(feat_folder, meal, savename), XYNeg, delimiter=",")
    # exit()
    # ##--------------------------------------------------------------------------------------



    XY = read_haar_feat_random_select_samples(meals, REC, WINSIZE, ratio = 5, use_seed = 1)
    # Y = XY[:,-1]

    XYPos =  XY[np.where(XY[:,-1]==1)[0],:]
    XYNeg =  XY[np.where(XY[:,-1]==0)[0],:]

    print('XYPos:',XYPos.shape)


    # # ##--------------------------------------------------------------------------------------
    # # #RANK THE FEATURE IMPORTANCE
    # build_strongclf_def_thres(XY, T = 10, mdlpath = './feat_rank/', debug = 1)
    # exit()
    # ##--------------------------------------------------------------------------------------


    # initialization

    XYPosTrn, XYPosTest = tt_split(XYPos, 0.7)
    XYNegTrn, XYNegTest = tt_split(XYNeg, 0.7)

    P = XYPosTrn
    N = XYNegTrn

    XYTrn = np.vstack((P,N))
    print('XYTrn shape:', XYTrn.shape)
    XYTest = np.vstack((XYPosTest,XYNegTest))
    print('XYTest shape:', XYTest.shape)

    cntFeats = XYTrn.shape[1]-1

    T = 20
    f = 0.3
    d = 0.7
    FTar = 0.1
    FList = []

    F = 1 #F0
    FPrev = 1
    D = 1 #D0

    clfThresList = []

    featList = [2, 9, 1, 65, 10, 59, 87, 61, 93, 86, 66, 89, 52, 51, 64, 83, 85, 3, 56, 58, 62, 0, 50, 77, 84, 95, 4, 63, 57, 291, 60, 92, 91, 96, 5, 11, 48, 49, 90, 112, 113, 203, 233, 237, 70, 88, 111, 241, 242, 17, 69, 74, 114, 278, 306, 270, 55, 98, 208, 73, 82, 284, 279, 12, 54, 67, 75, 109, 176, 199, 267, 272, 293, 304, 94, 177, 206, 239, 258, 261, 277, 316, 47, 68, 271, 169, 178, 230, 236, 238, 282, 37, 72, 97, 110, 116, 168, 265, 283, 288, 305, 308, 41, 71, 119, 194, 263, 102, 228, 244, 289, 299, 302, 303, 312, 314, 115, 181, 182, 202, 234, 240, 266, 313, 318, 322, 170, 185, 204, 269, 53, 117, 184, 188, 243, 257, 262, 209, 105, 189, 245, 286, 38, 118, 218, 315, 34, 78, 101, 106, 76, 81, 235, 268, 33, 43, 167, 205, 287, 292, 298, 19, 173, 224, 6, 7, 36, 211, 256, 28, 213, 251, 13, 79, 99, 126, 175, 223, 273, 260, 229, 290, 301, 18, 29, 32, 121, 123, 201, 212, 246, 252, 309, 39, 42, 100, 198, 264, 276, 294, 297, 311, 320, 27, 183, 191, 207, 324, 103, 104, 250, 317, 210, 307, 319, 389, 417, 190, 40, 120, 135, 219, 323, 395, 409, 423, 437, 26, 46, 127, 132, 136, 195, 214, 281, 404, 432, 14, 80, 145, 166, 196, 280, 400, 407, 428, 435, 193, 222, 134, 138, 197, 296, 310, 131, 153, 326, 416, 133, 172, 325, 122, 154, 22, 23, 107, 259, 328, 388, 394, 422, 20, 35, 45, 141, 329, 408, 410, 436, 438, 459, 474, 108, 174, 216, 374, 382, 399, 403, 406, 427, 431, 434, 458, 484, 490, 493, 31, 44, 142, 275, 285, 373, 492, 8, 15, 124, 220, 453, 456, 164, 137, 217, 321, 444, 449, 21, 128, 148, 151, 171, 215, 247, 255, 125, 156, 158, 187, 231, 232, 254, 295, 327, 371, 483, 487, 162, 165, 179, 359, 130, 253, 364, 368, 473, 494, 186, 381, 139, 147, 129, 221, 452, 457, 464, 489, 140, 146, 150, 152, 159, 274, 471, 25, 149, 157, 180, 353, 393, 415, 421, 443, 448, 455, 468, 478, 226, 331, 332, 339, 387, 390, 481, 161, 144, 248, 384, 386, 391, 392, 396, 411, 412, 413, 418, 143, 155, 200, 225, 227, 163, 385, 397, 398, 402, 419, 426, 430, 405, 433, 334, 340, 16, 160, 333, 341, 342, 424, 192, 330, 338, 372, 377, 378, 379, 380, 488, 491, 370, 383, 486, 30, 300, 358, 363, 367, 376, 249, 346, 482, 375, 472, 480, 24, 414, 420, 441, 336, 344, 439, 440, 335, 337, 343, 350, 401, 425, 429, 442, 445, 446, 447, 450, 451, 454, 345, 349, 351, 348, 462, 469, 476, 477, 479, 461, 463, 465, 466, 467, 470, 347, 352, 356, 357, 361, 366, 460, 475, 354, 355, 360, 362, 485, 369, 365]
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

        while F > FPrev*f and nFeats < cntFeats:
            
            nFeats = nFeats + 1
            print('nfeats:', nFeats)
            
            ######################################################################################
            #       build strong classifier with only first nFeats features in train set 
            ######################################################################################
            
            mdlpath = './fd_model_stage'+str(i)+'/'
        
            XYTrnNFeat = XYTrnUpdateWithTopNFeats(XYTrn, featList, nFeats)

    #       build stage with features selected
            build_strongclf_def_thres(XYTrnNFeat, T, mdlpath)
        
    #       evaluate cascaded classifier on validation set to determine F and D
            yRes, clfThres = load_strongclf_def_thres(XYTest, T, mdlpath)
            
            print("clfThres: ", clfThres)
            prec_pos, D, f1_pos, TPR, F, Specificity, MCC, CKappa, w_acc, cm = calc_cm_rcall(XYTest[:,-1], yRes)
            print("\nrecall_pos: ",D)
            print("false positive rate: ", F)
            
            if D > d and F < FPrev*f:
                clfThresList.append(clfThres)
                FList.append(F)
                
            elif D < d:
                thres = int(clfThres)
                
                while thres > 2 and D < d:
                    
                    thres = thres - 2
                    yRes = load_strongclf_adj_thres(XYTrnNFeat, T, thres, mdlpath)
                    
                    print("thres: ", thres)

                    # WHY USE XYTrnNFeat INSTEAD OF XYTEST
                    prec_pos, D, f1_pos, TPR, F, Specificity, MCC, CKappa, w_acc, cm = calc_cm_rcall(XYTrnNFeat[:,-1], yRes)

                    print("recall_pos: ",D)
                    print("false positive rate: ", F)
                    

                if D > d:
                    if F < FPrev*f:
                        clfThresList.append(thres)
                        FList.append(F)
                        print('Succeed!', '\n')
                    else:
                        print('Fail: F cannot be less than f when D is greater than or equal to d', '\n')
                else:
                    print('Fail, D cannot be greater than or equal to d', '\n')
                

            elif F > FPrev*f:
                    print('Specific case pursued came out: F > f !!! need to tune F, D, f, d', '\n')
    #       prediction for 
    #       evalutation
    #     update sample set
        XYTrn = update_trnset_with_FP_true_samples(XYTrn, yRes)


    ######################################################################################
    # 
    # 
    #       STAGE 2:
    # 
    # 
    ######################################################################################


    i = 1

    if i == 1: # build the 2nd stage
        # FPrev = FList[0]
        # F = FPrev
        FPrev = F

        print("Build stage 2:")

        while F > FPrev*f and nFeats < cntFeats:
            nFeats = nFeats + 1
            print('nFeats:', nFeats)
            
            ######################################################################################
            #       build strong classifier with only first nFeats features in train set 
            ######################################################################################
            
            mdlpath = './fd_model_stage'+str(i)+'/'
            
            XYTrnNFeat = XYTrnUpdateWithTopNFeats(XYTrn, featList, nFeats)# XYTrn is updated

    #       build stage with features selected
            build_strongclf_def_thres(XYTrnNFeat, T, mdlpath)
        
    #       evaluate cascaded classifier on validation set to determine F and D
            yRes = load_strongclf_adj_thres(XYTest, T, clfThresList[0], mdlpath)
            
            print("clfThres: ", clfThres)
            prec_pos, D, f1_pos, TPR, F, Specificity, MCC, CKappa, w_acc, cm = calc_cm_rcall(XYTest[:,-1], yRes)
            print("\nrecall_pos: ",D)
            print("false positive rate: ", F)
            
            if D > d and F < FPrev*f:
                clfThresList.append(clfThres)
                FList.append(F)
                
            elif D < d:
                thres = int(clfThres)
                
                while thres > 2 and D < d:
                    
                    thres = thres - 2
                    yRes = load_strongclf_adj_thres(XYTrnNFeat, T, thres, mdlpath)
                    print("thres: ", thres)

                    prec_pos, D, f1_pos, TPR, F, Specificity, MCC, CKappa, w_acc, cm = calc_cm_rcall(XYTrnNFeat[:,-1], yRes)

                    print("recall_pos: ",D)
                    print("false positive rate: ", F)
                    

                if D > d:
                    if F < FPrev*f:
                        clfThresList.append(thres)
                        FList.append(F)
                        print('Succeed!', '\n')
                    else:
                        print('Fail: F cannot be less than f when D is greater than or equal to d', '\n')
                else:
                    print('Fail, D cannot be greater than or equal to d', '\n')
                
                

            elif F > FPrev*f:
                    print('Specific case pursued came out: F > f !!! need to tune F, D, f, d', '\n')
                    
    #       prediction for 
            
    #       evalutation



    #     update sample set
    #     XYTrn = XYTrnUpdate(XYTrn, yRes, featList, nFeats, thres)



