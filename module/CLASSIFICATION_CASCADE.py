# 
# 
# [STEP 4 in readme]
# Classification
# 
# 

import os,re
import numpy as np
import pandas as pd
from beyourself_cascade_code_review_wo_PASDAClib import Cascaded





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




    for test_meal in meals:

        print("test_meal:")
        print(test_meal)

        LEVEL = 60

        
        """
        Test set
        """
        TEST_NEG_FEAT_FOLDER = os.path.join(OUT_FOLDER, test_meal, 'FEATURE', DEVICE, SENSOR, 'NEGATIVE')
        feat_file = 'dev_haar_type12_win'+str(FRAME_SIZE_SEC)+'_str'+str(STEP_SIZE_SEC)+'_all_len_fixed.csv'
        neg_df = pd.read_csv(os.path.join(TEST_NEG_FEAT_FOLDER, feat_file))
        neg_df['label'] = neg_df['label'].astype(bool).astype(int)

        TEST_POS_FEAT_FOLDER = os.path.join(OUT_FOLDER, test_meal, 'FEATURE', DEVICE, SENSOR, 'POSITIVE_LEVEL'+str(LEVEL))
        feat_file = 'dev_haar_type12_win'+str(FRAME_SIZE_SEC)+'_str'+str(STEP_SIZE_SEC)+'_overlapratio'+str(OVERLAP_RATIO)+'_all_len_fixed.csv'
        levelX_pos_df = pd.read_csv(os.path.join(TEST_POS_FEAT_FOLDER, feat_file))
        levelX_pos_df['label'] = levelX_pos_df['label'].astype(bool).astype(int)

        XYTest = pd.concat([neg_df, levelX_pos_df]).as_matrix()




        """
        Train set
        """
        LOPO_NEG_FEAT_FOLDER = os.path.join(LOPO_FOLDER, 'leave_'+test_meal, 'FEATURE', DEVICE, SENSOR, 'NEGATIVE')
        feat_file = 'dev_haar_type12_win'+str(FRAME_SIZE_SEC)+'_str'+str(STEP_SIZE_SEC)+'_all_len_fixed.csv'
        neg_df = pd.read_csv(os.path.join(LOPO_NEG_FEAT_FOLDER, feat_file))
        neg_df['label'] = neg_df['label'].astype(bool).astype(int)

        LOPO_POS_FEAT_FOLDER = os.path.join(LOPO_FOLDER, 'leave_'+test_meal, 'FEATURE', DEVICE, SENSOR, 'POSITIVE_LEVEL'+str(LEVEL))
        feat_file = 'dev_haar_type12_win'+str(FRAME_SIZE_SEC)+'_str'+str(STEP_SIZE_SEC)+'_overlapratio'+str(OVERLAP_RATIO)+'_all_len_fixed.csv'
        levelX_pos_df = pd.read_csv(os.path.join(LOPO_POS_FEAT_FOLDER, feat_file))
        levelX_pos_df['label'] = levelX_pos_df['label'].astype(bool).astype(int)

        XYTrn = pd.concat([neg_df, levelX_pos_df]).as_matrix()




        """
        Train set pos. vs neg.
        """

        XYNeg =  XYTrn[np.where(XYTrn[:,-1]==0)[0],:]
        XYPos =  XYTrn[np.where(XYTrn[:,-1]==1)[0],:]

        print(XYNeg.shape[0])
        print(XYPos.shape[0])




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















        











