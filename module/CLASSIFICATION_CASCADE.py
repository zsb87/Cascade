# 
# 
# [STEP 4 in readme]
# Classification
# 
# todo:
#   print  or draw all the predcition result on time aixs
# 

import os
import re
import numpy as np
import pandas as pd
import time
from beyourself_cascade_code_review_wo_PASDAClib import Cascaded
from util import create_folder, lprint






if __name__ == "__main__":

    logfile = './logfile1.txt'

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
    MEAL_FOLDER = os.path.join(OUT_DIR, subj, 'MEAL')

    LOPO_FOLDER = os.path.join(OUT_DIR, subj, 'MEAL_LOPO')




    for test_meal in meals:

        lprint(logfile, "TEST_MEAL:")
        lprint(logfile, test_meal)
        lprint(logfile, "...loading data")


        for MDL_LEVEL in LEVELS:
    
            start = time.time()

            """
            Train set
            """
            LOPO_NEG_FEAT_FOLDER = os.path.join(LOPO_FOLDER, 'leave_'+test_meal, 'FEATURE', DEVICE, SENSOR, 'NEGATIVE')
            feat_file = 'dev_haar_type12_win'+str(FRAME_SIZE_SEC)+'_str'+str(STEP_SIZE_SEC)+'_all_len_fixed.csv'
            neg_df = pd.read_csv(os.path.join(LOPO_NEG_FEAT_FOLDER, feat_file))
            neg_df['label'] = neg_df['label'].astype(bool).astype(int)

            LOPO_POS_FEAT_FOLDER = os.path.join(LOPO_FOLDER, 'leave_'+test_meal, 'FEATURE', DEVICE, SENSOR, 'POSITIVE_LEVEL'+str(MDL_LEVEL))
            feat_file = 'dev_haar_type12_win'+str(FRAME_SIZE_SEC)+'_str'+str(STEP_SIZE_SEC)+'_overlapratio'+str(OVERLAP_RATIO)+'_all_len_fixed.csv'
            levelX_pos_df = pd.read_csv(os.path.join(LOPO_POS_FEAT_FOLDER, feat_file))

            if len(levelX_pos_df) == 0:
                continue
            levelX_pos_df['label'] = levelX_pos_df['label'].astype(bool).astype(int)

            XYTrn = pd.concat([neg_df, levelX_pos_df]).as_matrix()


            """
            Train set pos. vs neg.
            """
            XYNeg =  XYTrn[np.where(XYTrn[:,-1]==0)[0],:]
            XYPos =  XYTrn[np.where(XYTrn[:,-1]==1)[0],:]

            print(XYNeg.shape[0])
            print(XYPos.shape[0])



            
            """
            Single level test set
            """

            TEST_LEVEL = MDL_LEVEL

            TEST_NEG_FEAT_FOLDER = os.path.join(MEAL_FOLDER, test_meal, 'FEATURE', DEVICE, SENSOR, 'NEGATIVE')
            feat_file = 'dev_haar_type12_win'+str(FRAME_SIZE_SEC)+'_str'+str(STEP_SIZE_SEC)+'_all_len_fixed.csv'
            neg_df = pd.read_csv(os.path.join(TEST_NEG_FEAT_FOLDER, feat_file))
            neg_df['label'] = neg_df['label'].astype(bool).astype(int)

            TEST_POS_FEAT_FOLDER = os.path.join(MEAL_FOLDER, test_meal, 'FEATURE', DEVICE, SENSOR, 'POSITIVE_LEVEL'+str(TEST_LEVEL))
            feat_file = 'dev_haar_type12_win'+str(FRAME_SIZE_SEC)+'_str'+str(STEP_SIZE_SEC)+'_overlapratio'+str(OVERLAP_RATIO)+'_all_len_fixed.csv'
            levelX_pos_df = pd.read_csv(os.path.join(TEST_POS_FEAT_FOLDER, feat_file))
            levelX_pos_df['label'] = levelX_pos_df['label'].astype(bool).astype(int)

            XYTest = pd.concat([neg_df, levelX_pos_df]).as_matrix()
            lprint(logfile, 'Single level test set: LEVEL', TEST_LEVEL)



            
            """
            Meal test set(all levels)
            """

            TEST_NEG_FEAT_FOLDER = os.path.join(MEAL_FOLDER, test_meal, 'FEATURE', DEVICE, SENSOR, 'NEGATIVE')
            feat_file = 'dev_haar_type12_win'+str(FRAME_SIZE_SEC)+'_str'+str(STEP_SIZE_SEC)+'_all_len_fixed.csv'
            neg_df = pd.read_csv(os.path.join(TEST_NEG_FEAT_FOLDER, feat_file))
            neg_df['label'] = neg_df['label'].astype(bool).astype(int)

            TEST_POS_FEAT_FOLDER = os.path.join(MEAL_FOLDER, test_meal, 'FEATURE', DEVICE, SENSOR, 'POSITIVE_LEVEL'+str(TEST_LEVEL))
            feat_file = 'dev_haar_type12_win'+str(FRAME_SIZE_SEC)+'_str'+str(STEP_SIZE_SEC)+'_overlapratio'+str(OVERLAP_RATIO)+'_all_len_fixed.csv'
            levelX_pos_df = pd.read_csv(os.path.join(TEST_POS_FEAT_FOLDER, feat_file))
            levelX_pos_df['label'] = levelX_pos_df['label'].astype(bool).astype(int)

            XYTest = pd.concat([neg_df, levelX_pos_df]).as_matrix()
            lprint(logfile, 'Meal test set(all level)')







            end = time.time()
            lprint(logfile, '...loading finish: ', end - start)
            lprint(logfile, "")
            

            """
            # cascaded classifier configuration
            # f: false positive rate
            # d: postive recall (detection rate)
            # T: number of iterations
            """

            stage_parameter=[(0.2, 0.92, 100),(0.2, 0.92, 100),(0.2, 0.92, 100)]
            split=0.7
            model_path="./modularized_cascade_model/"
            n_feats_max = XYTrn.shape[1]

            model = Cascaded(
                stage_parameter = stage_parameter,\
                split = split,\
                model_path = model_path,\
                n_feats_max = n_feats_max,\
                logfile = logfile
            )

            lprint(logfile, 'stage_parameter:', stage_parameter)
            lprint(logfile, 'split:', split)
            lprint(logfile, 'model_path:', model_path)
            lprint(logfile, "")



            # train/fit

            T_list, all_feat_list, n_feats_list, beta_list, thres_list, path_list = model.fit(XYTrn)

            # test: Single level test set

            F_list, D_list, F_final, D_final, y_res = model.test(XYTest, T_list, all_feat_list, n_feats_list, beta_list, thres_list, path_list)


            lprint(logfile, '===================== Single level test set =====================', D_list)
            lprint(logfile, 'Recall(postive) for every stage: ', D_list)
            lprint(logfile, 'False positive rate for every stage: ', F_list)
            lprint(logfile, 'Overall recall(postive): ', D_final)
            lprint(logfile, 'Overall false positive rate: ', F_final)

            # TEST_RESULT_FOLDER = os.path.join(MEAL_FOLDER, test_meal, 'RESULT', DEVICE, SENSOR, 'TESTDATA_LEVEL'+str(TEST_LEVEL)+'_MODEL_LEVEL'+str(MDL_LEVEL))
            # create_folder(TEST_RESULT_FOLDER)
            # savename = 'pred_dev_haar_type12_win'+str(FRAME_SIZE_SEC)+'_str'+str(STEP_SIZE_SEC)+'_overlapratio'+str(OVERLAP_RATIO)+'_all_len_fixed.txt'
            # np.savetxt(os.path.join(TEST_RESULT_FOLDER, savename), y_res, delimiter=",")




            # test: Meal test set(all levels)

            F_list, D_list, F_final, D_final, y_res = model.test(XYTest, T_list, all_feat_list, n_feats_list, beta_list, thres_list, path_list)

            lprint(logfile, '===================== Meal test set(all levels) =====================', D_list)
            lprint(logfile, 'Recall(postive) for every stage: ', D_list)
            lprint(logfile, 'False positive rate for every stage: ', F_list)
            lprint(logfile, 'Overall recall(postive): ', D_final)
            lprint(logfile, 'Overall false positive rate: ', F_final)

            TEST_RESULT_FOLDER = os.path.join(MEAL_FOLDER, test_meal, 'RESULT', DEVICE, SENSOR, 'TESTDATA_ALL_LEVEL_MODEL_LEVEL'+str(MDL_LEVEL))
            create_folder(TEST_RESULT_FOLDER)
            savename = 'pred_dev_haar_type12_win'+str(FRAME_SIZE_SEC)+'_str'+str(STEP_SIZE_SEC)+'_overlapratio'+str(OVERLAP_RATIO)+'_all_len_fixed.txt'
            np.savetxt(os.path.join(TEST_RESULT_FOLDER, savename), y_res, delimiter=",")






            lprint(logfile, "")
            lprint(logfile, "")
            lprint(logfile, "")
            lprint(logfile, "")



        











