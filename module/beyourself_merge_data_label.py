import matplotlib
import matplotlib.pyplot as plt
import pylab
from datetime import datetime, timedelta, time, date
from PASDAC.util import mark_class_period, hms_to_timedelta, rchop, create_folder, get_immediate_subdirectories

from beyourself.data.label import read_json, read_SYNC, read_ELAN, write_SYNC
from beyourself.core import maybe_create_folder

from shutil import copyfile
import numpy as np
import pandas as pd
import os
import json
import re
import time



# start = time.time()
# end = time.time()
# print(end - start)



# data hierarchical structure:
# 4 classes of data:
#     d1. known eating episode with label
#     d2. known eating episode without label (low video quality or not finished)
#     d3. known non-eating episode without label
#     d4. unknown eating/non-eating episode without label

# Trainset: d1 and d3
# Testset: d1, d3(feeding gesture recognition validation) and then d2(meal detection validation)
# Useless: d4

#              ?                E               NE             E
# video:  +++++++|++++++++++++++++++++|+++++++++++++++++++|++++++++|

#             d4             d1(d2)              d3         d1(d2)           d4
# data: ---------|--------------------|-------------------|--------|------------------------


# Todo: 
#    1. first use 'FG in d1' and 'nFG in d1' to train and test, then 
#    2. label start and end of d3




ABSOLUTE_TIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"



def read_label_summary(path):
    '''
    Read ELAN txt files into a pandas dataframe
    '''
    
    df = pd.read_csv(path)
    df = df[df['Use for Analysis(Y/N)'] == 'Y']
    
    return df


def adjust_annot_path(subj, subj_df, i, annot_file):
    if subj == 'P108' or subj == 'P112' or subj == 'P114':
        annot_file = rchop(annot_file,'.mov')

    annot_file = '/Users/shibozhang/Documents/Beyourself/beyourself-label/BeYourself/CLEAN/'+subj+'/FG_label/'+str(annot_file)+'.txt'
    if subj == 'P103':
        annot_file = '/Users/shibozhang/Documents/Beyourself/beyourself-label/BeYourself/CLEAN/'+subj+'/FG_label/'+subj_df['Label Path'].iloc[i]
    elif subj == 'P120':
        annot_file = '/Users/shibozhang/Documents/Beyourself/beyourself-label/BeYourself/CLEAN/'+subj+'/FG_label/'+rchop(subj_df['Label Path'].iloc[i],'.mov')+'.txt'
    elif subj == 'P107':
        annot_file = '/Users/shibozhang/Documents/Beyourself/beyourself-label/BeYourself/CLEAN/'+subj+'/FG_label/'+subj_df['Label Path'].iloc[i]+'.txt'

    return annot_file


def parse_videoname_from_path(string):
    print(string)
    RegExr = "DVR___\d+-\d+-\d+_\d+.\d+.\d+.AVI"
    m = re.search(RegExr, string)

    if m:
        return m.group()
    else:
        print('Video path error')
        return 0


def parse_timestamp_from_videoname(string):
    VIDEONAME_TIME_FORMAT = "%Y-%m-%d_%H.%M.%S"
    RegExr= '\d+-\d+-\d+_\d+.\d+.\d+'
    m = re.search(RegExr, string)

    if m:
        return datetime.strptime(m.group(), VIDEONAME_TIME_FORMAT)
    else:
        print('Video path error')
        return 0


def parse_timestamp_from_RelStart(string):
    STARTTIME_FORMAT = '%H:%M:%S'
    t = datetime.strptime(string, STARTTIME_FORMAT)
    delta = timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)
    return delta


def parse_timestamp_from_AbsStart(string):
    STARTTIME_FORMAT = '%m/%d/%y %H:%M:%S'
    return datetime.strptime(string, STARTTIME_FORMAT)



def conv_ELAN_to_SYNC(ELAN_df,video_starttime):
    SYNC_df = ELAN_df.copy()
    SYNC_df['start'] = ELAN_df['start'] + video_starttime
    SYNC_df['end'] = ELAN_df['end'] + video_starttime

    return SYNC_df


def truncate_df_index_dt(df, start, end, margin = 10):
    start_dt = start - timedelta(seconds=margin)
    end_dt = end + timedelta(seconds=margin)

    df = df[(df.index >= start_dt) & (df.index <= end_dt)]
    return df


def truncate_df_index_str(df, start, end):
    start_dt = datetime.strptime(start, ABSOLUTE_TIME_FORMAT) - timedelta(seconds=margin)
    end_dt = datetime.strptime(end, ABSOLUTE_TIME_FORMAT) + timedelta(seconds=margin)

    df = df[(df.index >= start_dt) & (df.index <= end_dt)]
    return df


def merge_data_label(annotDf, df_test):

    # 'fd' = 1 means feeding dominant hand
    # 'dd' = 1 means drinking dominant hand

    annot_fd = annotDf.loc[annotDf["label"]=="fd"]
    annot_dd = annotDf.loc[annotDf["label"]=="dd"]

    fd_st_list = list(annot_fd.start.tolist())
    fd_end_list = list(annot_fd.end.tolist())

    dd_st_list = list(annot_dd.start.tolist())
    dd_end_list = list(annot_dd.end.tolist())

    feeding_dur = []
    drinking_dur = []

    for n in range(len(fd_st_list)):
        feeding_dur.append([fd_st_list[n],fd_end_list[n]])

    for n in range(len(dd_st_list)):
        drinking_dur.append([dd_st_list[n],dd_end_list[n]])

    df_test_label = mark_class_period( df_test,'fd' , feeding_dur )
    df_test_label = mark_class_period( df_test_label,'dd' , drinking_dur )

    return df_test_label





subj = 'P116'#  'P103','P105','P107','P108','P109','P110','P111','P112','P114','P115','P116','P118',

first_time_flg = 1

all_feats_df = pd.DataFrame()

out_folder = '/Users/shibozhang/Documents/Beyourself/beyourself-label/BeYourself/TEMP/'



CLEAN_FOLDER = '/Volumes/SHIBO/BeYourself/BeYourself/CLEAN'
raw_video_folder = os.path.join(CLEAN_FOLDER, subj,'visualize/SYNC_meal')


PROC_FOLDER = '/Volumes/SHIBO/BeYourself/BeYourself/PROCESS'
proc_subj_folder = os.path.join(PROC_FOLDER, subj,'WRIST')


# meal_list = get_immediate_subdirectories(raw_video_folder)

# meals = ['m0824_1']
meals = ['m0815_1','m0815_2','m0821_1','m0821_2']


for meal in meals:    

    meal_folder = os.path.join(raw_video_folder, meal)
    label_file = os.path.join(meal_folder, 'labelFG_synced.json')

    save_folder = os.path.join(proc_subj_folder, meal, 'data_label')
    maybe_create_folder(save_folder)

    # read label file
    annotDf = read_SYNC(label_file)
    print(annotDf)

    # read acc file
    acc_file = os.path.join(meal_folder, 'accel.csv')
    acc_df = pd.read_csv(acc_file)
    acc_df['Unixtime'] = acc_df['Time']
    acc_df['Time'] = pd.to_datetime(acc_df['Time'],unit='ms',utc=True)
    acc_df = acc_df.set_index(['Time'])
    acc_df.index = acc_df.index.tz_localize('UTC').tz_convert('US/Central')

    # merge label file and acc file
    acc_label_df = merge_data_label(annotDf, acc_df)
    acc_label_df = truncate_df_index_dt(acc_label_df, annotDf.start.iloc[0], annotDf.end.iloc[-1])

    # save to PROC folder    
    acc_label_df.to_csv(os.path.join(save_folder, 'accel_label.csv'))



    # read gyro file
    gyr_file = os.path.join(meal_folder, 'gyro.csv')
    gyr_df = pd.read_csv(gyr_file)
    gyr_df['Unixtime'] = gyr_df['Time']
    gyr_df['Time'] = pd.to_datetime(gyr_df['Time'],unit='ms',utc=True)
    gyr_df = gyr_df.set_index(['Time'])
    gyr_df.index = gyr_df.index.tz_localize('UTC').tz_convert('US/Central')

    # merge label file and gyro file
    gyr_label_df = merge_data_label(annotDf, gyr_df)
    gyr_label_df = truncate_df_index_dt(gyr_label_df, annotDf.start.iloc[0], annotDf.end.iloc[-1])

    # save to PROC folder
    gyr_label_df.to_csv(os.path.join(save_folder, 'gyro_label.csv'))


