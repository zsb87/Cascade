import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, timedelta, time, date
from scipy.signal import savgol_filter
from get_haar import gen_feat_rec, gen_feat_rec_name
from beyourself.core.algorithm import *
from beyourself_event_eval import get_overlap_by_union_ratio
from wardmetrics.core_methods import eval_events
from PASDAC.util import pointwise2headtail, create_folder
import time


np.set_printoptions(precision=2)



def query_plot_acc(df, start, end, title='acc'):
    
    starttime = datetime.strptime(start, ABSOLUTE_TIME_FORMAT)
    endtime = datetime.strptime(end, ABSOLUTE_TIME_FORMAT)
    df = df[(df.Time > starttime) & (df.Time <= endtime)]

    df_accel = df[[ 'Time','accX', 'accY', 'accZ' ]]
    f = plt.figure(figsize=(15,5))
    styles1 = ['b-','r-','y-']
    df_accel.plot(style=styles1,ax=f.gca())
    plt.title(title, color='black')
    plt.savefig('/Volumes/SHIBO/BeYourself/BeYourself/PROCESS/P120/wrist/m0824_1/acc.png')
    
    return df_accel



def query_acc(df, start, end):
    starttime = datetime.strptime(start, ABSOLUTE_TIME_FORMAT)
    endtime = datetime.strptime(end, ABSOLUTE_TIME_FORMAT)
    df = df[(df.Time > starttime) & (df.Time <= endtime)]
    df_accel = df[[ 'Time','accX', 'accY', 'accZ' ]]
    return df_accel



ABSOLUTE_TIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f%z"

meal = 'm0824_1'


feat_folder = '/Volumes/SHIBO/BeYourself/BeYourself/PROCESS/P120/wrist/haar_feature/'
# file = '/Volumes/SHIBO/BeYourself/BeYourself/PROCESS/P108/WRIST_bymeal/0807meal1_part1/data_label/accel_label.csv'
acc_file = '/Volumes/SHIBO/BeYourself/BeYourself/PROCESS/P120/wrist/'+meal+'/data_label/accel_label.csv'
gyr_file = '/Volumes/SHIBO/BeYourself/BeYourself/PROCESS/P120/wrist/'+meal+'/data_label/gyro_label.csv'
# acc_df = pd.read_csv(acc_file)
# acc_df['Time'] = pd.to_datetime(acc_df['Time'])
# acc = query_plot_acc(acc_df, '2017-08-24 08:37:58.806-0500',  '2017-08-24 08:38:00.856-0500')


sensors = ["Accelerometer"]# "Gyroscope",

create_folder(os.path.join(feat_folder, meal))

# Set the frame and step size
FRAME_SIZE_SEC = 2
STEP_SIZE_SEC = float(FRAME_SIZE_SEC/2)
SIMPLING_RATE = 20



frame_size = int(FRAME_SIZE_SEC * SIMPLING_RATE)
step_size = int(STEP_SIZE_SEC * SIMPLING_RATE)

# save features in several single files for speed
SAVE_BATCH = 50*step_size

name_axes = ['accX', 'accY', 'accZ']


for sensor in sensors:
    
    first_time_flag = 1

    if sensor == 'Gyroscope':
        data_df = pd.read_csv(gyr_file)
        data_df = data_df[['Unixtime', 'rotX', 'rotY', 'rotZ', 'fd', 'dd']]

    if sensor == 'Accelerometer':
        data_df = pd.read_csv(acc_file)
        data_df = data_df[['Unixtime', 'accX', 'accY', 'accZ', 'fd', 'dd']]

    data_df['fd_dd'] = data_df['fd'] + data_df['dd'] 
    data_df = data_df[['accX', 'accY', 'accZ', 'fd_dd']]

    print(len(data_df))

    ground_truth_start_end = pointwise2headtail(data_df['fd_dd'].as_matrix()).tolist()
    print(ground_truth_start_end)

    M = data_df.as_matrix()

    TS = M[:,0]
    M = M[:,1:]

    # print ("")
    # print ("Shape of training data: " + str(M.shape))

    # Number of inputs
    number_of_inputs = M.shape[1]


    pos_examples_counter = 0
    neg_examples_counter = 0


    start = time.time()


    ##--------------------------------------------------------------------------------------
    # # SAVE COLUMN NAMES IN FILE
    # # Calculate features for frame
    # for counter in range(0,len(M)-frame_size,step_size):
    #     print(counter)
    #     savename = os.path.join(feat_folder, meal, 'feat_rec12_label_win'+str(FRAME_SIZE_SEC)+'_names.txt')
    #     fw = open(savename, 'w')
    #     for axis in range(number_of_inputs):
    #         V = M[counter:counter+frame_size, axis]
    #         names1 = gen_feat_rec_name(V, 4, 1)
    #         names2 = gen_feat_rec_name(V, 4, 2)
    #         names = names1 + names2
    #         print(names)
    #         for item in names:
    #             fw.write(name_axes[axis]+"_%s\n" % item)
    #     exit()
    ##--------------------------------------------------------------------------------------


    # Calculate features for frame
    for counter in range(0,len(M)-frame_size,step_size):
        print(counter)
        for axis in range(number_of_inputs):

            V = M[counter:counter+frame_size, axis]
            # V_D = savgol_filter(V, window_length=11, polyorder=2, deriv=0)
            # get the derivative/slope of signal
            Y_D = savgol_filter(V, window_length=11, polyorder=2, deriv=1)

            # get the harr feature 1, 2 and 3
            # H1 = gen_feat_rec(Y_D, stride = 4, rec = 1)
            # H2 = gen_feat_rec(Y_D, stride = 4, rec = 2)
            # H3 = gen_feat_rec(Y_D, stride = 4, rec = 3)

            if axis == 0:
                # ROW = np.hstack((H1, H2))
                # ROW = V.reshape((1,-1))
                ROW = Y_D.reshape((1,-1))
            else:
                # ROW = np.hstack((ROW, H1, H2))
                # ROW = np.hstack((ROW, V.reshape((1,-1))))
                ROW = np.hstack((ROW, Y_D.reshape((1,-1))))


        # # ----------------------------- Label -------------------------------------
        # # Add label
        # get start and end unixtime of segment, judge overlap score with event-based evaluation. if >75%, then 1 else 0

        current_frame_start_end = [(counter, counter+frame_size)]

        rate = get_overlap_by_union_ratio(ground_truth_start_end, current_frame_start_end)
        if rate > 0.75:
            label = 1
        else:
            label = 0

        R_T = np.hstack((ROW, np.array(rate).reshape((1,-1))))
        L_T = np.hstack((ROW, np.array(label).reshape((1,-1))))

        if counter%SAVE_BATCH == 0:
            if not first_time_flag:
                print(V_T2)
                print(V_T2.shape)
                # savename = 'feat_rec12_label_win'+str(FRAME_SIZE_SEC)+'_'+str(int(counter/SAVE_BATCH))+'.txt'
                # savename = 'accXYZ_label_win'+str(FRAME_SIZE_SEC)+'_'+str(int(counter/SAVE_BATCH))+'.txt'
                savename = 'devAccXYZ_label_win'+str(FRAME_SIZE_SEC)+'_'+str(int(counter/SAVE_BATCH))+'.txt'
                np.savetxt(os.path.join(feat_folder, meal, savename), V_T2, delimiter=",")

            V_T1 = R_T
            V_T2 = L_T
            end = time.time()
            print(end - start)

        else:
            V_T1 = np.vstack((V_T1, R_T))
            V_T2 = np.vstack((V_T2, L_T))


        first_time_flag = 0






# SAVE IN A SINGLE FILE
# if not os.path.exists(os.path.join(feat_folder, meal)):
#     os.makedirs(os.path.join(feat_folder, meal))

# np.savetxt(os.path.join(feat_folder, meal, 'feat_rate.txt'), V_T1, delimiter=",")




# print(acc)
# y = acc[['accX']]
# y = y.as_matrix().ravel()


# y_sm = savgol_filter(y, 5, 2)
# y_d_sm = savgol_filter(y, window_length=11, polyorder=2, deriv=1)

# print(len(y_d_sm))
# print(gen_feat_rec(y_d_sm))



