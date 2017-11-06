
# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd
import os
get_ipython().magic('matplotlib inline')
import matplotlib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from datetime import datetime, timedelta, time, date


# In[10]:

from os import listdir

def find_csv_filenames( path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]


# In[11]:

def add_padding_series(pw, padding_perc):
    pw = pw.astype(int)
    
    diff = np.hstack((pw,0)) - np.hstack((0,pw))
    head_ind_list = np.where( diff > 0 )[0]
    head_content_list = diff[head_ind_list]
    tail_ind_list = np.where( diff < 0 )[0]
    
    len_list = tail_ind_list - head_ind_list
    
    for i in range(len(head_ind_list)):
        
        pw[head_ind_list[i] - len_list[i]*padding_perc: head_ind_list[i]] = head_content_list[i]
        pw[tail_ind_list[i] : tail_ind_list[i] + len_list[i]*padding_perc] = head_content_list[i]

    return pw
    
    

    
def add_padding_df( df , cols , padding_perc ):
    
    for col in cols:
#         print(add_padding_series(df[col].as_matrix(), padding_perc)[1100:1200])
        df[col] = add_padding_series(df[col].as_matrix(), padding_perc)
        
    return df
        

# df = add_padding_df( df , ['fd', 'dd', 'fd_ind', 'dd_ind'] , 0.10 )


# In[122]:

def acc_deriv_sg(acc_df):
    
    acc_df['accX'] = savgol_filter(acc_df['accX'].as_matrix(), window_length=7, polyorder=2, deriv=1)    
    acc_df['accY'] = savgol_filter(acc_df['accY'].as_matrix(), window_length=7, polyorder=2, deriv=1)
    acc_df['accZ'] = savgol_filter(acc_df['accZ'].as_matrix(), window_length=7, polyorder=2, deriv=1)
    return acc_df


# In[123]:

def plot_acc_by_fgcategory_index(df, col, ind):
    mask = df[col] == ind
    df_mask = df.loc[mask]
    df_acc = df_mask[[ 'Time', 'accX', 'accY', 'accZ' ]]
    f = plt.figure(figsize=(15,5))
    styles1 = ['b-']

    df_acc.plot(style=styles1,ax=f.gca())
    print(str(ind)+'  start time:')
    print(str(df_acc['Time'].iloc[0]) )
    print(str(ind)+'  end time:')
    print(str(df_acc['Time'].iloc[-1]) )
    plt.title(str(ind)+'---'+str(str(df_acc['Time'].iloc[0])), color='black')
    
    return df_acc


# In[124]:

def plot_acc_deriv_by_fgcategory_index(df, col, ind):
    mask = df[col] == ind
    df_mask = df.loc[mask]
    df_acc = df_mask[[ 'Time', 'accX', 'accY', 'accZ' ]]
    f = plt.figure(figsize=(15,5))
    styles1 = ['b-']
    df_acc = acc_deriv_sg(df_acc)

    df_acc.plot(style=styles1,ax=f.gca())
    print(str(ind)+'  start time:')
    print(str(df_acc['Time'].iloc[0]) )
    print(str(ind)+'  end time:')
    print(str(df_acc['Time'].iloc[-1]) )
    plt.title(str(ind)+'---'+str(str(df_acc['Time'].iloc[0])), color='black')
    
    return df_acc


# In[125]:

def query_plot_acc(df, start, end, title):
    
    starttime = datetime.strptime(start, ABSOLUTE_TIME_FORMAT)
    endtime = datetime.strptime(end, ABSOLUTE_TIME_FORMAT)
    df = df[(df.Time > starttime) & (df.Time <= endtime)]

    df_accel = df[[ 'Time','accX', 'accY', 'accZ' ]]
    f = plt.figure(figsize=(15,5))
    styles1 = ['b-','r-','y-']
    df_accel.plot(style=styles1,ax=f.gca())
    plt.title(title, color='black')
    
    return df_accel


# In[119]:

import os

PROC_FOLDER = '/Volumes/SHIBO/BeYourself/BeYourself/PROCESS'
proc_subj_folder = os.path.join(PROC_FOLDER, 'P108/WRIST')
meal = '0807meal1_part1'
acc_label_file = os.path.join(proc_subj_folder, meal, 'data_label/accel_label_index.csv')


# fg_category = 'fd'
# fg_category = 'dd'

# acc_fd_folder = os.path.join(proc_subj_folder, meal, fg_category, 'acc')
# gyr_fd_folder = os.path.join(proc_subj_folder, meal, fg_category, 'gyr')

# def readFeedingGesture(fg_folder):
#     namelist = find_csv_filenames( fg_folder, suffix=".csv" )
#     n = len(namelist)
    
#     df_all = pd.DataFrame()
    
#     for i in range(n):
#         df = pd.read_csv( os.path.join(fg_folder , 'fg_' + str(i) + '.csv' ))
#         df['ind'] = i
#         df_all = pd.concat([df_all, df])
        
#     df_all = df_all[['ind' , 'Time', 'accX', 'accY', 'accZ']]
#     return df_all

# df = readFeedingGesture(acc_fd_folder)

df = pd.read_csv(acc_label_file)
for i in range(len(set(df.fd_ind))-1):
    plot_acc_by_fgcategory_index(df, 'fd_ind', i+1)
df = add_padding_df( df , ['fd', 'dd', 'fd_ind', 'dd_ind'] , 0.10 )


# In[120]:

for i in range(len(set(df.fd_ind))-1):
    
    plot_acc_by_fgcategory_index(df, 'fd_ind', i+1)
    



# In[126]:

for i in range(len(set(df.fd_ind))-1):

    plot_acc_deriv_by_fgcategory_index(df, 'fd_ind', i+1)
    


# In[24]:

file = '/Volumes/SHIBO/BeYourself/BeYourself/PROCESS/P108/WRIST/0807meal1_part1/data_label/accel_label.csv'
acc_df = pd.read_csv(file)
ABSOLUTE_TIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f%z"


acc_df['Time'] = pd.to_datetime(acc_df['Time'])
acc = query_plot_acc(acc_df, '2017-08-07 19:59:11.653000-0500',  '2017-08-07 19:59:17.709000-0500' , '')
# query_plot_acc(acc_df, '2017-08-07 20:04:27.061000-0500',  '2017-08-07 20:04:32.911000-0500' , '')


# In[244]:

file = '/Volumes/SHIBO/BeYourself/BeYourself/PROCESS/P108/WRIST/0807meal1_part1/data_label/accel_label.csv'
acc_df = pd.read_csv(file)
ABSOLUTE_TIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f%z"


acc_df['Time'] = pd.to_datetime(acc_df['Time'])
query_plot_acc(acc_df, '2017-08-07 19:59:37.970000-0500',  '2017-08-07 19:59:40.473000-0500' , '')
query_plot_acc(acc_df, '2017-08-07 20:05:05.997000-0500',  '2017-08-07 20:05:08.301000-0500' , '')
query_plot_acc(acc_df, '2017-08-07 20:00:58.879000-0500',  '2017-08-07 20:01:04.832000-0500' , '')


# In[80]:

from scipy.signal import savgol_filter
np.set_printoptions(precision=2)


# In[82]:

y = acc[['accX']]
y = y.as_matrix().ravel()
x = len(y)
x = np.array(list(range(x)))/20
# print(x)
# print(y)

y_sm = savgol_filter(y, 5, 2)


# In[92]:

plt.plot(y)


# In[93]:

y_sm = savgol_filter(y, window_length=5, polyorder=2, deriv=0)
plt.plot(y_sm)


# In[94]:

y_sm = savgol_filter(y, window_length=7, polyorder=2, deriv=0)
plt.plot(y_sm)


# In[95]:

y_sm = savgol_filter(y, window_length=9, polyorder=2, deriv=0)
plt.plot(y_sm)


# In[96]:

y_sm = savgol_filter(y, window_length=11, polyorder=2, deriv=0)
plt.plot(y_sm)


# In[97]:

y_sm = savgol_filter(y, window_length=5, polyorder=4, deriv=0)
plt.plot(y_sm)


# In[90]:

y_d_sm = savgol_filter(y, window_length=7, polyorder=2, deriv=1)
plt.plot(y_d_sm)


# In[100]:

y_d_sm = savgol_filter(y, window_length=5, polyorder=2, deriv=1)
plt.plot(y_d_sm)


# In[101]:

y_d_sm = savgol_filter(y, window_length=9, polyorder=2, deriv=1)
plt.plot(y_d_sm)


# In[102]:

y_d_sm = savgol_filter(y, window_length=11, polyorder=2, deriv=1)
plt.plot(y_d_sm)


# In[127]:

len(y_d_sm)


# ## create derivative integral signal

# In[131]:

def gen_integ(y):
    y_int = np.zeros(y.size)
    y_int[0] = y[0]
    for i in range(1, y.size):
        y_int[i] = y_int[i-1] + y[i]
        
    return y_int


# ## harr-like feature 1: one part

# In[146]:

def gen_feat_1rec(y, stride = 4):
    l = y.size
    y_integ = gen_integ(y)
    # l = 120, stride = 4
    feats = np.zeros(435)
    n = 0
    for i in range(0, l, stride):
        for j in range(i+stride, l, stride):
            feats[n] = y_integ[j] - y_integ[i]
            n = n + 1
    return feats


# In[134]:

y_d_sm_int = gen_integ(y_d_sm)


# In[136]:

plt.plot(y_d_sm_int)


# In[168]:

plt.plot(gen_feat_1rec(y_d_sm))


# In[162]:

gen_feat_1rec(y_d_sm)


# ## harr-like feature 2: two part

# In[159]:

def gen_feat_2rec(y, stride = 4):
    l = y.size
    y_integ = gen_integ(y)
    # l = 120, stride = 4
    feats = np.zeros(4060)
    n = 0
    for i in range(0, l, stride):
        for j in range(i+stride, l, stride):
            for k in range(j+stride, l, stride):
                tmp1 = y_integ[j] - y_integ[i]
                tmp2 = y_integ[k] - y_integ[j]
                feats[n] = tmp1 - tmp2*(j - i)/(k - j)
                n = n + 1
    return feats


# In[172]:

plt.plot(gen_feat_2rec(y_d_sm))


# ## harr-like feature 3: three part

# In[161]:

# def gen_feat_3rec(y, stride = 4):
#     l = y.size
#     y_integ = gen_integ(y)
#     # l = 120, stride = 4
#     feats = np.zeros(4060)
#     n = 0
#     for i in range(0, l, stride):
#         for j in range(i+stride, l, stride):
#             for k in range(j+stride, l, stride):
#                 for m in range(j+stride, l, stride):
#                     tmp1 = y_integ[j] - y_integ[i]
#                     tmp2 = y_integ[k] - y_integ[j]
#                     feats[n] = tmp1 - tmp2*(j - i)/(k - j)
#                     n = n + 1
#     return feats


# In[171]:

plt.plot(np.hstack((gen_feat_1rec(y_d_sm), gen_feat_2rec(y_d_sm))))


# ## non-feeding gesture Harr-like features

# In[182]:

n_acc = query_plot_acc(acc_df, '2017-08-07 19:59:15.709000-0500','2017-08-07 19:59:21.759000-0500', '')


# In[183]:

n_acc


# In[184]:

ny = n_acc[['accX']]
ny = ny.as_matrix().ravel()


# In[186]:

ny_d_sm = savgol_filter(ny, window_length=11, polyorder=2, deriv=1)
plt.plot(ny_d_sm)


# In[189]:

plt.plot(gen_feat_1rec(ny_d_sm))


# In[191]:

plt.plot(gen_feat_2rec(ny_d_sm))


# In[192]:

df


# # resampling

# In[194]:

df['Time'] = pd.to_datetime(df['Time'])


# In[205]:

resmp_df = df.copy()


# In[206]:

resmp_df = resmp_df.set_index('Time')


# In[207]:

resmp_df
# data.index = pd.to_datetime(data.index, unit='s')


# In[ ]:

resmp_df.resample("20ms", fill_method="ffill")


# In[ ]:

tmpdf =pd.DataFrame.to_clipboard()


# In[ ]:

tmpdf


# In[ ]:



