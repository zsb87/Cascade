import os
import pandas as pd
from datetime import datetime, timedelta, time, date
from shutil import copyfile
from PASDAC.util import mark_class_period, hms_to_timedelta, rchop, create_folder, get_immediate_subdirectories
from beyourself.core.util import datetime_to_epoch
from beyourself.data.label import read_ELAN


ABSOLUTE_TIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"

ABSOLUTE_TIME_FORMAT_NO_MILLISEC = "%Y-%m-%d %H:%M:%S"


def read_label_summary(path):
    '''
    Read ELAN txt files into a pandas dataframe
    '''
    
    df = pd.read_csv(path)
    # df = df[(df['Use for Analysis(Y/N)'] == 'Y')]# & (df['WristOn(Y/N)'] == 'Y')
    
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



def gen_sensor_hour_file_name(year, month, day, hour):
    return str('{:02d}'.format(month))+'-'+str('{:02d}'.format(day))+'-'+str(AbsStart.year)[2:]+'_'+str('{:02d}'.format(hour))+'.csv'



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



def parse_timedelta_from_RelStart(string):
    STARTTIME_FORMAT = '%H:%M:%S'
    t = datetime.strptime(string, STARTTIME_FORMAT)
    delta = timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)
    return delta



def parse_datetime_from_RelStart(string):
    STARTTIME_FORMAT = '%H:%M:%S'
    t = datetime.strptime(string, STARTTIME_FORMAT)
    return t



def parse_timestamp_from_AbsStart(string):
    STARTTIME_FORMAT_WO_CENTURY = '%m/%d/%y %H:%M:%S'
    STARTTIME_FORMAT_W_CENTURY = '%m/%d/%Y %H:%M:%S'
    try:
        try:
            dt = datetime.strptime(string, STARTTIME_FORMAT_WO_CENTURY)
        except: 
            dt = datetime.strptime(string, ABSOLUTE_TIME_FORMAT_NO_MILLISEC)

    except:
        dt = datetime.strptime(string, STARTTIME_FORMAT_W_CENTURY)

    return dt



def conv_ELAN_to_SYNC(ELAN_df,video_starttime):
    SYNC_df = ELAN_df.copy()
    SYNC_df['start'] = ELAN_df['start'] + video_starttime
    SYNC_df['end'] = ELAN_df['end'] + video_starttime

    return SYNC_df





subjs = ['P114']#  'P103','P105','P107','P108','P109','P110','P111','P112','P114','P115','P116','P118',

# i_list = list

first_time_flg = 1


for subj in subjs:

    print(subj)
    # read start time and end time for a meal
    path = '/Users/shibozhang/Documents/Beyourself/beyourself-label/BeYourself/CLEAN/'+subj+'/labeling_'+subj+'.csv'
    subj_df = read_label_summary(path)
    subj_df['acc reliability'] = 0
    subj_df['gyr reliability'] = 0
    # subj_df['necklace reliability'] = 0



    for i in range(len(subj_df)):#

        # if 1:

        meal = subj_df['ID'].iloc[i]
        print(meal)

        # CREATE NEW MEAL FOLDER
        MEAL_FOLDER = '/Volumes/SHIBO/BeYourself/BeYourself/CLEAN/'+subj+'/visualize/SYNC_meal/'+meal
        create_folder(MEAL_FOLDER)

        video_file = subj_df['Path'].iloc[i]
        annot_file = adjust_annot_path(subj, subj_df, i, video_file)


        starttime_ext = hms_to_timedelta(subj_df['RelStart'].iloc[i])-timedelta(seconds = 10)\
             if hms_to_timedelta(subj_df['RelStart'].iloc[i])>timedelta(seconds = 10) \
             else timedelta(seconds = 0)
        endtime_ext = hms_to_timedelta(subj_df['RelEnd'].iloc[i])+timedelta(seconds = 10)


        # COPY ELAN LABEL FILE TO NEW MEAL FOLDER
        create_folder(os.path.join(MEAL_FOLDER,'label_ELAN'))
        # copyfile(annot_file, os.path.join(MEAL_FOLDER, 'label_ELAN', meal+'.txt'))
        # ELAN_annot_df = read_ELAN(annot_file)        
        # ELAN_annot_df = ELAN_annot_df[(ELAN_annot_df['start']>starttime_ext) & (ELAN_annot_df['end']<endtime_ext)]
        # ELAN_annot_df = ELAN_annot_df.sort_values('start')
        # ELAN_annot_df.to_csv(os.path.join(MEAL_FOLDER, meal+'.txt'), index =None)



        # COPY DATA TO NEW MEAL FOLDER
        # GET ABS_STARTTIME AND ABS_ENDTIME:
        RelStart_td = hms_to_timedelta(subj_df['RelStart'].iloc[i])
        RelEnd_td = hms_to_timedelta(subj_df['RelEnd'].iloc[i])
        print(subj_df['AbsStart'].iloc[i])
        AbsStart = parse_timestamp_from_AbsStart(subj_df['AbsStart'].iloc[i])
        AbsEnd = AbsStart + RelEnd_td - RelStart_td

        acc_path = os.path.join('/Volumes/SHIBO/BeYourself/BeYourself/TEMP/WRIST-git',\
                             subj, 'Accelerometer')
        gyr_path = os.path.join('/Volumes/SHIBO/BeYourself/BeYourself/TEMP/WRIST-git',\
                             subj, 'Gyroscope')
        # necklace_path = os.path.join('/Volumes/SHIBO/BeYourself/BeYourself/CLEAN',\
        #                      subj, 'necklace/data')

        data_start_file_name = gen_sensor_hour_file_name(AbsStart.year, AbsStart.month, AbsStart.day, AbsStart.hour)
        data_end_file_name = gen_sensor_hour_file_name(AbsEnd.year, AbsEnd.month, AbsEnd.day, AbsEnd.hour)

            
        if data_start_file_name == data_end_file_name: # if this meal is in one hour's file:
            try:
                acc_df = pd.read_csv(os.path.join(acc_path, data_start_file_name))
                gyr_df = pd.read_csv(os.path.join(gyr_path, data_start_file_name))

            except:
                print('FILE NOT EXSIT: ',os.path.join(acc_path, data_start_file_name))
                print('OR: ',os.path.join(gyr_path, data_start_file_name))
                continue
                # necklace_df = pd.read_csv(os.path.join(necklace_path, data_start_file_name))
        else:
                # CONCAT data_start_file_name and data_end_file_name
            try:
                acc_df1 = pd.read_csv(os.path.join(acc_path, data_start_file_name))
                acc_df2 = pd.read_csv(os.path.join(acc_path, data_end_file_name))
                acc_df = pd.concat((acc_df1, acc_df2))

                gyr_df1 = pd.read_csv(os.path.join(gyr_path, data_start_file_name))
                gyr_df2 = pd.read_csv(os.path.join(gyr_path, data_end_file_name))
                gyr_df = pd.concat((gyr_df1, gyr_df2))

            except:
                print('FILE NOT EXSIT: ',os.path.join(acc_path, data_start_file_name))
                print('OR: ',os.path.join(acc_path, data_end_file_name))
                continue
                # necklace_df1 = pd.read_csv(os.path.join(necklace_path, data_start_file_name))
                # necklace_df2 = pd.read_csv(os.path.join(necklace_path, data_end_file_name))
                # necklace_df = pd.concat((necklace_df1, necklace_df2))


        print('(AbsStart):', (AbsStart))
        print('datetime_to_epoch(AbsStart):', datetime_to_epoch(AbsStart))
        print('RelStart_td:', RelStart_td)
        print('RelStart_td.total_seconds()*1000:', RelStart_td.total_seconds()*1000)



        # generate acc data for SYNC tool and SAVE DATA FILE IN MEAL FOLDER
        acc_df = acc_df.sort_values('Time')
        acc_df['ELAN_time'] = acc_df['Time'] - datetime_to_epoch(AbsStart) + RelStart_td.total_seconds()*1000
        acc_df['ELAN_time'].iloc[0] = 0
        acc_df['ELAN_time'] = acc_df['ELAN_time'].astype(int)
        create_folder(os.path.join(MEAL_FOLDER,'data_wrist'))
        # acc_df.to_csv(os.path.join(MEAL_FOLDER,'data_wrist', 'acc_'+data_start_file_name), index = None)

        # ACC WRIST DATA RELIABILITY
        length = len(acc_df[(acc_df.Time > datetime_to_epoch(AbsStart)) & (acc_df.Time <= datetime_to_epoch(AbsEnd))])
        create_folder(os.path.join(MEAL_FOLDER,'data_wrist_reliability'))
        with open(os.path.join(MEAL_FOLDER,'data_wrist_reliability', 'acc data usability.txt'), 'w') as text_file:
            r = 50*float(length)/((datetime_to_epoch(AbsEnd)- datetime_to_epoch(AbsStart)))
            text_file.write('acc sensor reliability:'+str(r))
        subj_df['acc reliability'].iloc[i] = r
        print(r)



        # generate gyr data for SYNC tool, SAVE DATA FILE IN MEAL FOLDER
        gyr_df = gyr_df.sort_values('Time')
        gyr_df['ELAN_time'] = gyr_df['Time'] - datetime_to_epoch(AbsStart) + RelStart_td.total_seconds()*1000
        gyr_df['ELAN_time'].iloc[0] = 0
        gyr_df['ELAN_time'] = gyr_df['ELAN_time'].astype(int)
        create_folder(os.path.join(MEAL_FOLDER,'data_wrist'))
        # gyr_df.to_csv(os.path.join(MEAL_FOLDER,'data_wrist', 'gyr_'+data_start_file_name), index = None)
        
        # GYR WRIST DATA RELIABILITY
        length = len(gyr_df[(gyr_df.Time > datetime_to_epoch(AbsStart)) & (gyr_df.Time <= datetime_to_epoch(AbsEnd))])
        create_folder(os.path.join(MEAL_FOLDER,'data_wrist_reliability'))
        with open(os.path.join(MEAL_FOLDER,'data_wrist_reliability', 'gyr data usability.txt'), 'w') as text_file:
            r = 50*float(length)/((datetime_to_epoch(AbsEnd)- datetime_to_epoch(AbsStart)))
            text_file.write('gyr sensor reliability:'+str(r))
        subj_df['gyr reliability'].iloc[i] = r
        print(r)



            # # generate necklace data for SYNC tool, SAVE DATA FILE IN MEAL FOLDER
            # necklace_df = necklace_df.sort_values('Time')
            # necklace_df['ELAN_time'] = necklace_df['Time'] - datetime_to_epoch(AbsStart) + RelStart_td.total_seconds()*1000
            # necklace_df['ELAN_time'].iloc[0] = 0
            # necklace_df['ELAN_time'] = necklace_df['ELAN_time'].astype(int)
            # create_folder(os.path.join(MEAL_FOLDER,'data_necklace'))
            # # necklace_df.to_csv(os.path.join(MEAL_FOLDER, 'data_necklace', 'necklace_'+data_start_file_name), index = None)

            # # Necklace DATA RELIABILITY
            # length = len(necklace_df[(necklace_df.Time > datetime_to_epoch(AbsStart)) & (necklace_df.Time <= datetime_to_epoch(AbsEnd))])
            # create_folder(os.path.join(MEAL_FOLDER,'data_necklace_reliability'))
            # with open(os.path.join(MEAL_FOLDER,'data_necklace_reliability', 'necklace data usability.txt'), 'w') as text_file:
            #     r = 50*float(length)/((datetime_to_epoch(AbsEnd)- datetime_to_epoch(AbsStart)))
            #     text_file.write('necklace sensor reliability:'+str(r))
            # subj_df['necklace reliability'].iloc[i] = r


    subj_df.to_csv('/Users/shibozhang/Documents/Beyourself/beyourself-label/BeYourself/CLEAN/'+subj+'/labeling_'+subj+'_for_analysis_wrist_reliability.csv',index = None)

