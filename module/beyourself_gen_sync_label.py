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


# data hierarchical structure:
# 4 classes of data:
#     d1. known eating episode with label
#     d2. known eating episode without label (low video quality or not finished)
#     d3. known non-eating episode without label
#     d4. unknown eating/non-eating episode without label

# Trainset: d1 and d3
# Testset: d1, d3(feeding gesture recognition validation) and then d2(meal detection validation)
# Useless: d4

#              ?                E                   NE             E
# video:  +++++++|++++++++++++++++++++|+++++++++++++++++++|++++++++|

#             d4             d1(d2)              d3             d1(d2)           d4
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





subjs = ['P120']#  'P103','P105','P107','P108','P109','P110','P111','P112','P114','P115','P116','P118',

first_time_flg = 1

all_feats_df = pd.DataFrame()



for subj in subjs:
    print(subj)
    path = '/Users/shibozhang/Documents/Beyourself/beyourself-label/BeYourself/CLEAN/'+subj+'/labeling_'+subj+'.csv'
    

    subj_df = read_label_summary(path)

    existing_meals = get_immediate_subdirectories('/Volumes/SHIBO/BeYourself/BeYourself/CLEAN/P120/visualize/SYNC_necklace/')

    for i in range(len(subj_df)):

        meal = subj_df['ID'].iloc[i]
        print(meal)

        annot_file = subj_df['Path'].iloc[i]

        starttime = hms_to_timedelta(subj_df['RelStart'].iloc[i])-timedelta(seconds = 10)\
             if hms_to_timedelta(subj_df['RelStart'].iloc[i])>timedelta(seconds = 10) \
             else timedelta(seconds = 0)
        endtime = hms_to_timedelta(subj_df['RelEnd'].iloc[i])+timedelta(seconds = 10)
        
        annot_file = adjust_annot_path(subj, subj_df, i, annot_file)

        AbsStart = parse_timestamp_from_AbsStart(subj_df['AbsStart'].iloc[i])
        RelStart = parse_timestamp_from_RelStart(subj_df['RelStart'].iloc[i])


        ELAN_annot_df = read_ELAN(annot_file)
        ELAN_annot_df = ELAN_annot_df[(ELAN_annot_df['start']>starttime) & (ELAN_annot_df['end']<endtime)]
        ELAN_annot_df = ELAN_annot_df.sort_values('start')

        # print(ELAN_annot_df)

        video_starttime = AbsStart - RelStart
        SYNC_annot_df = conv_ELAN_to_SYNC(ELAN_annot_df,video_starttime)

        SYNC_FG_path = '/Volumes/SHIBO/BeYourself/BeYourself/CLEAN/'+subj+'/visualize/SYNC_FG/'+meal
        create_folder(SYNC_FG_path)
        print(SYNC_annot_df)

        write_SYNC(SYNC_annot_df, os.path.join(SYNC_FG_path,'labelFG.json'))

        SYNC_necklace_path = '/Volumes/SHIBO/BeYourself/BeYourself/CLEAN/'+subj+'/visualize/SYNC_necklace/'+meal
        
        if meal in existing_meals:
            copyfile(os.path.join(SYNC_FG_path,'labelFG.json'), \
                 os.path.join(SYNC_necklace_path,'labelFG.json'))


        



