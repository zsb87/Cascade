import os,re
import numpy as np
import pandas as pd
import random
from sklearn.metrics import *





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


def lprint(logfile, *argv): # for python version 3

    """ 
    Function description: 
    ----------
        Save output to log files and print on the screen.

    Function description: 
    ----------
        var = 1
        lprint('log.txt', var)
        lprint('log.txt','Python',' code')

    Parameters
    ----------
        logfile:                 the log file path and file name.
        argv:                    what should 
        
    Return
    ------
        none

    """

    # argument check
    if len(argv) == 0:
        print('Err: wrong usage of func lprint().')
        sys.exit()

    argAll = argv[0] if isinstance(argv[0], str) else str(argv[0])
    for arg in argv[1:]:
        argAll = argAll + (arg if isinstance(arg, str) else str(arg))
    
    print(argAll)

    with open(logfile, 'a') as out:
        out.write(argAll + '\n')



def calc_cm_rcall(y_test, y_pred):

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

    