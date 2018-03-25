# 
# 
# [STEP 2 in readme]
# Using label column from label_level.csv, generate feature file, with an extra column of len_level taking the length. 
# 
# 

import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, timedelta, time, date
from scipy.signal import savgol_filter
from get_haar import gen_feat_rec, gen_feat_rec_name
from gen_stat_FFT_feat import gen_stat_FFT_feat, genFeatsNames
from beyourself_event_eval import get_overlap_by_union_ratio, get_overlap_by_winsize_ratio
import time, datetime


np.set_printoptions(precision=2)



def get_overlap(a, b):
    '''
    Given two pairs a and b,
    find the intersection between them
    could be either number or datetime objects
    '''

    tmp = min(a[1], b[1]) - max(a[0], b[0])
    
    if isinstance(tmp, timedelta):
        zero_value = timedelta(seconds=0)
    else:
        zero_value = 0
    
    return max(zero_value, tmp)


def get_union(a, b):
    '''
    Given two pairs a and b,
    assume a and b have overlap
    find the union between them
    could be either number or datetime objects
    '''

    tmp = max(a[1], b[1]) - min(a[0], b[0])
    # if a and b have no overlap
    # todo
    return tmp


def pointwise2headtail(pointwise):
    '''
    return the index of the first non-zero element and the index of the last non-zero element
    >>>pointwise2headtail([0,0,1,1,0])
    >>>[[2 3]]

    '''
    diff = np.concatenate((pointwise[:],np.array([0]))) - np.concatenate((np.array([0]),pointwise[:]))
    ind_head = np.where(diff == 1)[0]
    ind_tail = np.where(diff == -1)[0]-1

    headtail = np.vstack((ind_head, ind_tail)).T;

    return headtail


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


def query_plot_acc(df, start, end, title='acc'):
    
    starttime = datetime.strptime(start, ABSOLUTE_TIME_FORMAT)
    endtime = datetime.strptime(end, ABSOLUTE_TIME_FORMAT)
    df = df[(df.Time > starttime) & (df.Time <= endtime)]

    df_accel = df[[ 'Time','accX', 'accY', 'accZ' ]]
    f = plt.figure(figsize=(15,5))
    styles1 = ['b-','r-','y-']
    df_accel.plot(style=styles1,ax=f.gca())
    plt.title(title, color='black')
    plt.savefig('/Volumes/SHIBO/BeYourself/BeYourself/PROCESS/'+subj+'/wrist/m0824_1/acc.png')
    
    return df_accel


def query_acc(df, start, end):
    starttime = datetime.strptime(start, ABSOLUTE_TIME_FORMAT)
    endtime = datetime.strptime(end, ABSOLUTE_TIME_FORMAT)
    df = df[(df.Time > starttime) & (df.Time <= endtime)]
    df_accel = df[[ 'Time','accX', 'accY', 'accZ' ]]
    return df_accel


def map_gt_to_fixed_length(ht, LEVEL):
    '''
    expend [head, tail] to [head_ext, tail_ext] while satisfying 'tail_ext-head_ext+1' is the closest element in the LEVEL list
    >>>ht=[20, 82], LEVEL=[10, 20, 40, 60, 80, 120, 160]
    >>>print(map_gt_to_fixed_length(ht))
    >>>[11, 90]. (right_append = 9, left_append = 8)

    # when TARGET_LEVEL-l = even, append '(TARGET_LEVEL-l)/2' to left and '(TARGET_LEVEL-l)/2' to right
    # when TARGET_LEVEL-l = odd, append '(TARGET_LEVEL-l)/2 - 1/2' to left and '(TARGET_LEVEL-l)/2 + 1/2' to right

    '''
    head = ht[0]
    tail = ht[1]
    l = tail - head + 1

    LEVEL_arr = np.array(LEVEL)
    right = np.array(np.where(LEVEL_arr>=l))

    if right.shape[1] == 0:
        lprint('test.txt',datetime.datetime.now(),'\n','Longer FG than 6 seconds -  head & tail:', head, ', ', tail)
        return 0
    else:
        target_ind = right[0][0]

        TARGET_LEVEL = LEVEL[target_ind]

        right_append = int((TARGET_LEVEL - l + 1)/2)
        left_append = TARGET_LEVEL - l - right_append
        ht_ext = [head-right_append, tail+left_append]

        return ht_ext



def gen_stack_feature(M, FRAME_SIZE, STEP_SIZE, FEATURE_CATEGORY, SAMPLING_RATE=0):


    for counter in range(0,len(M)-FRAME_SIZE,STEP_SIZE):

        # statistical and FFT features:
        if FEATURE_CATEGORY == 4:
    
            V = M[counter:counter+FRAME_SIZE, :]
            H = gen_stat_FFT_feat(V, SAMPLING_RATE, stride=4)
            ROW = H.reshape((1,-1))


        elif FEATURE_CATEGORY == 0:

            for axis in range(number_of_inputs):

                V = M[counter:counter+FRAME_SIZE, axis]

                # raw signal of gyro and accel.  filename: accXYZ_label_win, gyrXYZ_label_win
            
                V_D = savgol_filter(V, window_length=11, polyorder=2, deriv=0)
                if axis == 0:
                    ROW = V_D.reshape((1,-1))
                else:
                    ROW = np.hstack((ROW, V_D.reshape((1,-1))))


            # derivative of gyro and accel. filename: devGyrXYZ_label_winX, devAccXYZ_label_winX
        elif FEATURE_CATEGORY == 1: 

            for axis in range(number_of_inputs):

                V = M[counter:counter+FRAME_SIZE, axis]
                # get the derivative/slope of signal
                Y_D = savgol_filter(V, window_length=11, polyorder=2, deriv=1)
                if axis == 0:
                    ROW = Y_D.reshape((1,-1))
                else:
                    ROW = np.hstack((ROW, Y_D.reshape((1,-1))))


            # haar feature of gyro and accel. filename: feat_type12_label_winX
        elif FEATURE_CATEGORY == 2:

            for axis in range(number_of_inputs):
                
                V = M[counter:counter+FRAME_SIZE, axis]

                Y_D = savgol_filter(V, window_length=11, polyorder=2, deriv=0)
                # get the harr feature 1, 2 and 3
                H1 = gen_feat_rec(Y_D, stride=4, type=1)
                H2 = gen_feat_rec(Y_D, stride=4, type=2)
                # H3 = gen_feat_rec(Y_D, stride=4, type=3)

                if axis == 0:
                    ROW = np.hstack((H1, H2))
                else:
                    ROW = np.hstack((ROW, H1, H2))


            # haar feature of derivative of gyro and accel. filename: feat_dev_type12_label_winX
        elif FEATURE_CATEGORY == 3:

            for axis in range(number_of_inputs):
                
                V = M[counter:counter+FRAME_SIZE, axis]

                Y_D = savgol_filter(V, window_length=11, polyorder=2, deriv=1)
                # get the harr feature 1, 2 and 3
                H1 = gen_feat_rec(Y_D, stride=4, type=1)
                H2 = gen_feat_rec(Y_D, stride=4, type=2)
                # H3 = gen_feat_rec(Y_D, stride=4, type=3)

                if axis == 0:
                    ROW = np.hstack((H1, H2))
                else:
                    ROW = np.hstack((ROW, H1, H2))


        if counter == 0:
            F_T = ROW
        
        else:
            F_T = np.vstack((F_T, ROW))


    return F_T



def gen_stack_label(N, FRAME_SIZE, STEP_SIZE, OVERLAP_RATIO, ground_truth_start_end, fill_in='max', g_or_geq=0):

    """

    g_or_geq:   0 means great than OVERLAP_RATIO
                1 means great then or equal to OVERLAP_RATIO

    """

    label_list = []

    for counter in range(0,len(N)-FRAME_SIZE,STEP_SIZE):
        current_win_start_end = [(counter, counter+FRAME_SIZE)]
        rate = get_overlap_by_winsize_ratio(ground_truth_start_end, current_win_start_end)

        if g_or_geq:
            # param to be tuned
            if rate >= OVERLAP_RATIO:
                if fill_in == 1:
                    label = 1
                elif fill_in == 'max':
                    label = max(N[counter:counter+FRAME_SIZE])
                    # print(label)
            else:
                label = 0

        else:
            # param to be tuned
            if rate > OVERLAP_RATIO:
                if fill_in == 1:
                    label = 1
                elif fill_in == 'max':
                    label = max(N[counter:counter+FRAME_SIZE])
                    # print(label)
            else:
                label = 0

        label_list.append(label)

    L_T = np.array(label_list)

    return L_T




# def filter(df):
#     flt_para = 10
#     df.accx = pd.rolling_mean(df.accx, flt_para)
#     df.accy = pd.rolling_mean(df.accy, flt_para)
#     df.accz = pd.rolling_mean(df.accz, flt_para)
    
#     df.rotx = pd.rolling_mean(df.rotx, flt_para)
#     df.roty = pd.rolling_mean(df.roty, flt_para)
#     df.rotz = pd.rolling_mean(df.rotz, flt_para)
    
#     df.pitch_deg = pd.rolling_mean(df.pitch_deg, flt_para)
#     df.roll_deg = pd.rolling_mean(df.roll_deg, flt_para)
    
#     df = df.dropna()
#     return df





if __name__ == "__main__":


    ABSOLUTE_TIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f%z"

    settings = {}
    settings["TIMEZONE"] = 'Etc/GMT+6'


    DEVICE = 'WRIST'

    SENSOR = 'ACC_GYR'

    subj = 'P120'

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


    DATA_DIR = '/Volumes/Seagate/SHIBO/BeYourself-Structured/DATA_LABEL'
    DATA_FOLDER = os.path.join(DATA_DIR, subj, 'MEAL')


    OUT_DIR =  '/Volumes/Seagate/SHIBO/BeYourself-Structured/OUTPUT'
    OUT_FOLDER = os.path.join(OUT_DIR, subj, 'MEAL')


    FRAME_SIZE_SEC = 2 # CORRESPONDING TO LEVEL 40
    SAMPLING_RATE = 20
    STEP_SIZE_SEC = float(FRAME_SIZE_SEC/2)
    OVERLAP_RATIO = 0.75
    FRAME_SIZE = int(FRAME_SIZE_SEC * SAMPLING_RATE)
    STEP_SIZE = int(STEP_SIZE_SEC * SAMPLING_RATE)


    # save features in several single files for speed up
    # SAVE_BATCH = 50*STEP_SIZE


    # FEATURE_CATEGORY = 0 # raw signal of gyro and accel.  filename: accXYZ_label_win, gyrXYZ_label_win
    # FEATURE_CATEGORY = 1 # derivative of gyro and accel. filename: devGyrXYZ_label_winX, devAccXYZ_label_winX
    # FEATURE_CATEGORY = 2 # haar feature of raw gyro and accel. filename: 
    # FEATURE_CATEGORY = 3 # haar feature of derivative of gyro and accel. filename: feat_type12_label_winX
    FEATURE_CATEGORY = 4 # statistical and FFT feature. filename: feat_stat_FFT_label_winX
        



    for meal in meals:    

        print(meal)

        DATA_MEAL_FOLDER = os.path.join(DATA_FOLDER, meal, 'data_label')
        data_df = pd.read_csv(os.path.join(DATA_MEAL_FOLDER, 'acc_gyr_label_'+meal+'.csv'))

        FEAT_FOLDER = os.path.join(OUT_FOLDER, meal, 'FEATURE', DEVICE, SENSOR)
        create_folder(FEAT_FOLDER)

        label_df = pd.read_csv(os.path.join(FEAT_FOLDER, 'label_level.csv'))



        # Set the frame and step size

        LEVELS = [10, 20, 40, 60, 80, 120, 160]



        """
        1. load data_label.
        """

        data_df['f_d'] = data_df['fd'] + data_df['dd'] + data_df['ft'] + data_df['dt']
        data_df = data_df[['Unixtime','accX','accY','accZ','rotX','rotY','rotZ','f_d']]
        data_df['f_d'] = data_df['f_d'].astype(bool).astype(int)
        print(len(data_df))

        ground_truth_start_end = pointwise2headtail(data_df['f_d'].as_matrix()).tolist()
        # print('ground_truth_start_end:')
        # print(ground_truth_start_end)


        """
        2. go through every feeding gesture, for every FG, map to one fixed length close to its length
        """

        ground_truth_start_end_fixed = []

        for ht in ground_truth_start_end:
            ht_ext = map_gt_to_fixed_length(ht, LEVELS)
            if ht_ext: # if ht_ext is not 0
                ground_truth_start_end_fixed.append(ht_ext)
                # print(ht_ext[1]-ht_ext[0]+1)

        print('ground_truth_start_end_fixed:')
        print(ground_truth_start_end_fixed)




        M = data_df.as_matrix()

        # TS = M[:,0]
        M = M[:,1:-1]

        print ("Shape of training data: " + str(M.shape))

        # Number of inputs/signal channels
        number_of_inputs = M.shape[1]




        """
        3. SAVE COLUMN NAMES IN FILE
        """

        if FEATURE_CATEGORY == 3: # haar feature of derivative of gyro and accel.

            savename = os.path.join(FEAT_FOLDER, 'dev_haar_type12_win'+str(FRAME_SIZE_SEC)+'_str'+str(STEP_SIZE_SEC)+'_names.txt')
            fw = open(savename, 'w')

            for axis in range(number_of_inputs):
                counter = 0
                V = M[counter:counter+FRAME_SIZE, axis]
                names1 = gen_feat_rec_name(data=V, stride=4, type=1)
                names2 = gen_feat_rec_name(data=V, stride=4, type=2)
                names = names1 + names2
                for item in names:
                    name_axes = ['accX','accY','accZ','rotX','rotY','rotZ']
                    fw.write(name_axes[axis]+"_%s\n" % item)
            # fw.write("label\n")


        elif FEATURE_CATEGORY == 4:
            sensor_list = ['accX','accY','accZ','rotX','rotY','rotZ']
            names = genFeatsNames(sensor_list)
            savename = os.path.join(FEAT_FOLDER, 'stat_FFT_label_win'+str(FRAME_SIZE_SEC)+'_str'+str(STEP_SIZE_SEC)+'_names.txt')
            fw = open(savename, 'w')
            for name in names:
                fw.write("%s\n" % name)
            # fw.write("label\n")




        """
        4. generate feature file, with an extra column of len_level taking the length. 
        """

        # Calculate features for frame

        F_T = gen_stack_feature(M, FRAME_SIZE, STEP_SIZE, FEATURE_CATEGORY)



        """
        5. SAVE IN A SINGLE FILE
        todo: OVERLAP_RATIO should be removed from the name as the feat is independent of OVERLAP_RATIO
        """

        if FEATURE_CATEGORY == 3:
            savename = 'dev_haar_type12_win'+str(FRAME_SIZE_SEC)+'_str'+str(STEP_SIZE_SEC)+'_overlapratio'+str(OVERLAP_RATIO)+'_all_len_fixed.txt'
            np.savetxt(os.path.join(FEAT_FOLDER, savename), F_T, delimiter=",")

        elif FEATURE_CATEGORY == 4:
            savename = 'stat_FFT_label_win'+str(FRAME_SIZE_SEC)+'_str'+str(STEP_SIZE_SEC)+'_overlapratio'+str(OVERLAP_RATIO)+'_all_len_fixed.txt'
            np.savetxt(os.path.join(FEAT_FOLDER, savename), F_T, delimiter=",")





        # """
        # 6. Add label - OV_label(OVerlap label)
        #     get start and end unixtime of segment, judge overlap score with event-based evaluation. if >75%, then max else 0
            
        # """
        # N = label_df['f_d'].as_matrix()
        
        # L_T = gen_stack_label(N, FRAME_SIZE, STEP_SIZE, OVERLAP_RATIO, ground_truth_start_end_fixed, fill_in='max', g_or_geq=1)

        # savename = 'label_win'+str(FRAME_SIZE_SEC)+'_str'+str(STEP_SIZE_SEC)+'_overlapratio'+str(OVERLAP_RATIO)+'_geq_fillin_max_all_len_fixed.txt'
        # np.savetxt(os.path.join(FEAT_FOLDER, savename), L_T, delimiter=",")





        # """
        # 7. Add label - AZ_label(Any siZe label)
        #     get start and end unixtime of segment, judge overlap score with event-based evaluation. if >0%, then max else 0
            
        # """
        # N = label_df['f_d'].as_matrix()
        
        # L_T = gen_stack_label(N, FRAME_SIZE, STEP_SIZE, \
        #                         OVERLAP_RATIO=0, \
        #                         ground_truth_start_end=ground_truth_start_end_fixed, \
        #                         fill_in='max', \
        #                         g_or_geq=0)
 
        # savename = 'label_win'+str(FRAME_SIZE_SEC)+'_str'+str(STEP_SIZE_SEC)+'_overlapratio'+str(0)+'_g_fillin_max_all_len_fixed.txt'
        # np.savetxt(os.path.join(FEAT_FOLDER, savename), L_T, delimiter=",")


        # exit()




























            # if counter%SAVE_BATCH == 0:

            #     if not first_time_flg:

            #         # raw signal of gyro and accel.  filename: accXYZ_label_win, gyrXYZ_label_win
                    

            #         if FEATURE_CATEGORY == 0:
            #             if SENSOR == 'Gyroscope':
            #                 savename = 'gyr_label_win'+str(FRAME_SIZE_SEC)+'_str'+str(STEP_SIZE_SEC)+'_overlapratio'+str(OVERLAP_RATIO)+'_part'+str(i_batch)+'_len_fixed.txt'
            #             elif SENSOR == 'Accelerometer':
            #                 savename = 'acc_label_win'+str(FRAME_SIZE_SEC)+'_str'+str(STEP_SIZE_SEC)+'_overlapratio'+str(OVERLAP_RATIO)+'_part'+str(i_batch)+'_len_fixed.txt'
            #             elif SENSOR == 'ACC_GYR':
            #                 savename = 'label_win'+str(FRAME_SIZE_SEC)+'_str'+str(STEP_SIZE_SEC)+'_overlapratio'+str(OVERLAP_RATIO)+'_part'+str(i_batch)+'_len_fixed.txt'


            #         # derivative of gyro and accel. filename: devGyrXYZ_label_winX, devAccXYZ_label_winX
            #         elif FEATURE_CATEGORY == 1:
            #             if SENSOR == 'Gyroscope':
            #                 savename = 'dev_gyr_label_win'+str(FRAME_SIZE_SEC)+'_str'+str(STEP_SIZE_SEC)+'_overlapratio'+str(OVERLAP_RATIO)+'_part'+str(i_batch)+'_len_fixed.txt'
            #             elif SENSOR == 'Accelerometer':
            #                 savename = 'dev_acc_label_win'+str(FRAME_SIZE_SEC)+'_str'+str(STEP_SIZE_SEC)+'_overlapratio'+str(OVERLAP_RATIO)+'_part'+str(i_batch)+'_len_fixed.txt'
            #             elif SENSOR == 'ACC_GYR':
            #                 savename = 'dev_label_win'+str(FRAME_SIZE_SEC)+'_str'+str(STEP_SIZE_SEC)+'_overlapratio'+str(OVERLAP_RATIO)+'_part'+str(i_batch)+'_len_fixed.txt'


            #         # savename = 'feat_type12_label_win'+str(FRAME_SIZE_SEC)+'_str'+str(STEP_SIZE_SEC)+'_overlapratio'+str(OVERLAP_RATIO)+'_part'+str(i_batch)+'_len_fixed.txt'
            #         elif FEATURE_CATEGORY == 2:
            #             if SENSOR == 'Gyroscope':
            #                 savename = 'dev_gyr_type12_label_win'+str(FRAME_SIZE_SEC)+'_str'+str(STEP_SIZE_SEC)+'_overlapratio'+str(OVERLAP_RATIO)+'_part'+str(i_batch)+'_len_fixed.txt'
            #             elif SENSOR == 'Accelerometer':
            #                 savename = 'dev_acc_type12_label_win'+str(FRAME_SIZE_SEC)+'_str'+str(STEP_SIZE_SEC)+'_overlapratio'+str(OVERLAP_RATIO)+'_part'+str(i_batch)+'_len_fixed.txt'
            #             elif SENSOR == 'ACC_GYR':
            #                 savename = 'dev_type12_label_win'+str(FRAME_SIZE_SEC)+'_str'+str(STEP_SIZE_SEC)+'_overlapratio'+str(OVERLAP_RATIO)+'_part'+str(i_batch)+'_len_fixed.txt'

            #         # savename = 'feat_type12_label_win'+str(FRAME_SIZE_SEC)+'_str'+str(STEP_SIZE_SEC)+'_overlapratio'+str(OVERLAP_RATIO)+'_part'+str(i_batch)+'_len_fixed.txt'
            #         elif FEATURE_CATEGORY == 4:
            #             if SENSOR == 'Gyroscope':
            #                 savename = 'gyr_stat_FFT_label_win'+str(FRAME_SIZE_SEC)+'_str'+str(STEP_SIZE_SEC)+'_overlapratio'+str(OVERLAP_RATIO)+'_part'+str(i_batch)+'_len_fixed.txt'
            #             elif SENSOR == 'Accelerometer':
            #                 savename = 'acc_stat_FFT_label_win'+str(FRAME_SIZE_SEC)+'_str'+str(STEP_SIZE_SEC)+'_overlapratio'+str(OVERLAP_RATIO)+'_part'+str(i_batch)+'_len_fixed.txt'
            #             elif SENSOR == 'ACC_GYR':
            #                 savename = 'stat_FFT_label_win'+str(FRAME_SIZE_SEC)+'_str'+str(STEP_SIZE_SEC)+'_overlapratio'+str(OVERLAP_RATIO)+'_part'+str(i_batch)+'_len_fixed.txt'

            #         np.savetxt(os.path.join(FEAT_FOLDER, savename), V_T2, delimiter=",")

            #     # V_T1 = R_T
            #     V_T2 = L_T
            #     end = time.time()
            #     print(end - start)

            # else:
            #     # V_T1 = np.vstack((V_T1, R_T))
            #     V_T2 = np.vstack((V_T2, L_T))

            # first_time_flg = 0




        # SAVE THE LAST FILE
        # if FEATURE_CATEGORY == 2:
        #     savename = 'dev_type12_label_win'+str(FRAME_SIZE_SEC)+'_str'+str(STEP_SIZE_SEC)+'_overlapratio'+str(OVERLAP_RATIO)+'_part'+str(i_batch+1)+'_len_fixed.txt'
        #     np.savetxt(os.path.join(FEAT_FOLDER, savename), V_T2, delimiter=",")
        # elif FEATURE_CATEGORY == 4:
        #     savename = 'stat_FFT_label_win'+str(FRAME_SIZE_SEC)+'_str'+str(STEP_SIZE_SEC)+'_overlapratio'+str(OVERLAP_RATIO)+'_part'+str(i_batch+1)+'_len_fixed.txt'
        #     np.savetxt(os.path.join(FEAT_FOLDER, savename), V_T2, delimiter=",")








