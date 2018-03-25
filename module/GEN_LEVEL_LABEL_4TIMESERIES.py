# 
# 
# [STEP 1 in readme]
# GENERATE A CSV WITH ONLY ONE COLUMN, label with the fixed length of the feeding gesture
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
from beyourself_event_eval import get_overlap_by_union_ratio
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
    
    NOTE: The non-zero element has to be 1.

    >>>pointwise2headtail([0,0,1,1,0])
    >>>[[2 3]]

    '''
    diff = np.concatenate((pointwise[:],np.array([0]))) - np.concatenate((np.array([0]),pointwise[:]))
    ind_head = np.where(diff == 1)[0]
    ind_tail = np.where(diff == -1)[0]-1
    # print(len(ind_tail))
    # print(len(ind_head))

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
    >>>ht = [20, 82]
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
    # print(right)

    if right.shape[1] == 0:
        lprint('test.txt',datetime.datetime.now(),': Longer FG than 6 seconds -  head & tail:', head, ', ', tail)
        return 0
    else:
        target_ind = right[0][0]

        TARGET_LEVEL = LEVEL[target_ind]

        right_append = int((TARGET_LEVEL - l + 1)/2)
        left_append = TARGET_LEVEL - l - right_append
        ht_ext = [head-right_append, tail+left_append]

        return ht_ext


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
    """
    Generate label_level.csv for each meal

    label_level.csv: feeding gesture data using the value of length instead of 1
    0, 0, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 0, 0, 0, 0, 0

    """


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
    'm0829_2'
    # 'm0830_1'\ # label error, [[200, 224], [338, 359], [524, 22820]]
    ]



    first_time_flg = 1

    all_feats_df = pd.DataFrame()

    DATA_DIR = '/Volumes/Seagate/SHIBO/BeYourself-Structured/DATA_LABEL'
    DATA_FOLDER = os.path.join(DATA_DIR, subj, 'MEAL')

    activity_dict = ['fd','ft','dd','dt']

    OUT_DIR =  '/Volumes/Seagate/SHIBO/BeYourself-Structured/OUTPUT'
    OUT_FOLDER = os.path.join(OUT_DIR, subj, 'MEAL')
    create_folder(OUT_FOLDER)





    for meal in meals:    

        print(meal)

        meal_folder = os.path.join(DATA_FOLDER, meal, 'data_label')
        data_df = pd.read_csv(os.path.join(meal_folder, 'acc_gyr_label_'+meal+'.csv'))

        FEAT_FOLDER = os.path.join(OUT_FOLDER, meal, 'FEATURE', DEVICE, SENSOR)
        create_folder(FEAT_FOLDER)

        # Set the frame and step size

        LEVEL = [10, 20, 40, 60, 80, 120, 160]


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
            ht_ext = map_gt_to_fixed_length(ht, LEVEL)
            if ht_ext: # if ht_ext is not 0
                ground_truth_start_end_fixed.append(ht_ext)
                length = ht_ext[1]-ht_ext[0]+1

                print(length)

                # change the label value in data_df['f_d']
                data_df['f_d'].iloc[ht_ext[0]:ht_ext[1]+1] = length
                if length == 40:
                    print(ht_ext)



        """
        3. save data_df['f_d'] in file 
        """
        
        label_df = data_df['f_d']
        label_df.to_csv(os.path.join(FEAT_FOLDER,'label_level.csv'),index=None, header=True)
        
        print('ground_truth_start_end_fixed:')
        print(ground_truth_start_end_fixed)



        print('')
        print('')
        print('')
        print('')
        print('')










