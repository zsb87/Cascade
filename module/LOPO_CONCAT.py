# 
# 
# [STEP 3 in readme]
# Gen training set:
#         When generating trainset for level X, 
#         remove all the positive and negative samples with len_level attribute other than X. 
#     Eg: when building level-20 classifier, remove all the samples with len_level 10,40,60,80,100,120,160
# 
# 



import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
import pickle
from six import string_types
import collections





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


def slid_win_remove_intersection(df, label_col):
    df1 = df.copy()

    # df[label_col] = df[label_col].astype(bool).astype(int)
    return df1[df1[label_col]==0]


def slid_win_keep_intersection(df, label_col, LEVEL):
    df1 = df.copy()

    return df1[df1[label_col]==LEVEL]






if __name__ == "__main__":

    subj = 'P120'
    DEVICE = 'WRIST'
    SENSOR = 'ACC_GYR'

    meals = [\
    'm0824_1',\
    'm0824_2',\
    'm0826_2',\
    'm0826_5',\
    'dk0826_2',\
    'm0826_6',\
    'm0827_2',\
    'm0828_1',\
    'm0828_2',\
    'm0828_3',\
    'm0828_4',\
    'm0829_2',\
    # 'm0830_1'\ # label error, [[200, 224], [338, 359], [524, 22820]]
    ]

    LEVELS = [10, 20, 40, 60, 80, 120, 160]

    FRAME_SIZE_SEC = 2 # CORRESPONDING TO LEVEL 40
    SAMPLING_RATE = 20
    STEP_SIZE_SEC = float(FRAME_SIZE_SEC/2)
    OVERLAP_RATIO = 0.75
    FRAME_SIZE = int(FRAME_SIZE_SEC * SAMPLING_RATE)
    STEP_SIZE = int(STEP_SIZE_SEC * SAMPLING_RATE)

    FEATURE_CATEGORY = 3 # haar feature of derivative of gyro and accel. filename: feat_type12_label_winX
        

    OUT_DIR =  '/Volumes/Seagate/SHIBO/BeYourself-Structured/OUTPUT'
    OUT_FOLDER = os.path.join(OUT_DIR, subj, 'MEAL')

    LOPO_FOLDER = os.path.join(OUT_DIR, subj, 'MEAL_LOPO')




    for exclude_meal in meals:

        """----------------------------------------------------------------------------------


        LOPO test set

        
        ----------------------------------------------------------------------------------"""


        FEAT_FOLDER = os.path.join(OUT_FOLDER, exclude_meal, 'FEATURE', DEVICE, SENSOR)

        # load label - OV_label(OVerlap label)
        label_file = 'label_win'+str(FRAME_SIZE_SEC)+'_str'+str(STEP_SIZE_SEC)+'_overlapratio'+str(OVERLAP_RATIO)+'_geq_fillin_max_all_len_fixed.txt'
        ov_label_df = pd.read_csv(os.path.join(FEAT_FOLDER, label_file), names=['label'])


        # load label - AZ_label(Any siZe label)
        label_file = 'label_win'+str(FRAME_SIZE_SEC)+'_str'+str(STEP_SIZE_SEC)+'_overlapratio'+str(0)+'_g_fillin_max_all_len_fixed.txt'
        az_label_df = pd.read_csv(os.path.join(FEAT_FOLDER, label_file), names=['label'])


        # load feature names
        names_file_path = os.path.join(FEAT_FOLDER, 'dev_haar_type12_win'+str(FRAME_SIZE_SEC)+'_str'+str(STEP_SIZE_SEC)+'_names.txt')
        with open(names_file_path, "r") as ins:
            names = []
            for line in ins:
                names.append(line)

        # load feature
        feat_file = 'dev_haar_type12_win'+str(FRAME_SIZE_SEC)+'_str'+str(STEP_SIZE_SEC)+'_overlapratio'+str(OVERLAP_RATIO)+'_all_len_fixed.txt'
        feat_df = pd.read_csv(os.path.join(FEAT_FOLDER, feat_file), names=names)


        """
        remove any intersection and save:
            that is, remove all the instances intersected with feeding gesture of any length with any overlap
        """
        
        # merge feature and label
        neg_df = pd.concat([feat_df, az_label_df], axis=1)

        neg_df = slid_win_remove_intersection(neg_df,'label')

        TEST_NEG_FEAT_FOLDER = os.path.join(FEAT_FOLDER, 'NEGATIVE')
        create_folder(TEST_NEG_FEAT_FOLDER)
        
        feat_file = 'dev_haar_type12_win'+str(FRAME_SIZE_SEC)+'_str'+str(STEP_SIZE_SEC)+'_all_len_fixed.csv'
        neg_df.to_csv(os.path.join(TEST_NEG_FEAT_FOLDER, feat_file), index=None)




        """
        add level X and save
            that is, for level X, add the instances intersected with feeding gesture with over OVERLAP_RATIO overlap
        """

        # merge feature and label
        pos_df = pd.concat([feat_df, ov_label_df], axis=1)

        for LEVEL in LEVELS:
            levelX_pos_df = slid_win_keep_intersection(pos_df,'label',LEVEL)


            TEST_POS_FEAT_FOLDER = os.path.join(FEAT_FOLDER, 'POSITIVE_LEVEL'+str(LEVEL))
            create_folder(TEST_POS_FEAT_FOLDER)

            feat_file = 'dev_haar_type12_win'+str(FRAME_SIZE_SEC)+'_str'+str(STEP_SIZE_SEC)+'_overlapratio'+str(OVERLAP_RATIO)+'_all_len_fixed.csv'
            levelX_pos_df.to_csv(os.path.join(TEST_POS_FEAT_FOLDER, feat_file), index=None)








        # """----------------------------------------------------------------------------------


        # LOPO training set

        
        # ----------------------------------------------------------------------------------"""

        # df_concat = []
        # az_label_df_concat = []
        # ov_label_df_concat = []

        # for meal in meals:

        #     if meal == exclude_meal:
        #         continue

        #     FEAT_FOLDER = os.path.join(OUT_FOLDER, meal, 'FEATURE', DEVICE, SENSOR)

        #     # load label - OV_label(OVerlap label)
        #     label_file = 'label_win'+str(FRAME_SIZE_SEC)+'_str'+str(STEP_SIZE_SEC)+'_overlapratio'+str(OVERLAP_RATIO)+'_geq_fillin_max_all_len_fixed.txt'
        #     ov_label_df = pd.read_csv(os.path.join(FEAT_FOLDER, label_file), names=['label'])


        #     # load label - AZ_label(Any siZe label)
        #     label_file = 'label_win'+str(FRAME_SIZE_SEC)+'_str'+str(STEP_SIZE_SEC)+'_overlapratio'+str(0)+'_g_fillin_max_all_len_fixed.txt'
        #     az_label_df = pd.read_csv(os.path.join(FEAT_FOLDER, label_file), names=['label'])


        #     # load feature names
        #     names_file_path = os.path.join(FEAT_FOLDER, 'dev_haar_type12_win'+str(FRAME_SIZE_SEC)+'_str'+str(STEP_SIZE_SEC)+'_names.txt')
        #     with open(names_file_path, "r") as ins:
        #         names = []
        #         for line in ins:
        #             names.append(line)

        #     # load feature
        #     feat_file = 'dev_haar_type12_win'+str(FRAME_SIZE_SEC)+'_str'+str(STEP_SIZE_SEC)+'_overlapratio'+str(OVERLAP_RATIO)+'_all_len_fixed.txt'
        #     df = pd.read_csv(os.path.join(FEAT_FOLDER, feat_file), names=names)


        #     df_concat.append(df)
        #     az_label_df_concat.append(az_label_df)
        #     ov_label_df_concat.append(ov_label_df)



        # feat_df = pd.concat(df_concat)
        # az_label_df = pd.concat(az_label_df_concat)
        # ov_label_df = pd.concat(ov_label_df_concat)



        # """
        # remove any intersection and save:
        #     that is, remove all the instances intersected with feeding gesture of any length with any overlap
        # """
        
        # # merge feature and label
        # neg_df = pd.concat([feat_df, az_label_df], axis=1)

        # neg_df = slid_win_remove_intersection(neg_df,'label')

        # LOPO_NEG_FEAT_FOLDER = os.path.join(LOPO_FOLDER, 'leave_'+exclude_meal, 'FEATURE', DEVICE, SENSOR, 'NEGATIVE')
        # create_folder(LOPO_NEG_FEAT_FOLDER)
        
        # feat_file = 'dev_haar_type12_win'+str(FRAME_SIZE_SEC)+'_str'+str(STEP_SIZE_SEC)+'_all_len_fixed.csv'
        # neg_df.to_csv(os.path.join(LOPO_NEG_FEAT_FOLDER, feat_file), index=None)




        # """
        # add level X and save
        #     that is, for level X, add the instances intersected with feeding gesture with over OVERLAP_RATIO overlap
        # """

        # # merge feature and label
        # pos_df = pd.concat([feat_df, ov_label_df], axis=1)

        # for LEVEL in LEVELS:
        #     levelX_pos_df = slid_win_keep_intersection(pos_df,'label',LEVEL)


        #     LOPO_POS_FEAT_FOLDER = os.path.join(LOPO_FOLDER, 'leave_'+exclude_meal, 'FEATURE', DEVICE, SENSOR, 'POSITIVE_LEVEL'+str(LEVEL))
        #     create_folder(LOPO_POS_FEAT_FOLDER)

        #     feat_file = 'dev_haar_type12_win'+str(FRAME_SIZE_SEC)+'_str'+str(STEP_SIZE_SEC)+'_overlapratio'+str(OVERLAP_RATIO)+'_all_len_fixed.csv'
        #     levelX_pos_df.to_csv(os.path.join(LOPO_POS_FEAT_FOLDER, feat_file), index=None)







