import os
import re
import csv
import matplotlib
import numpy as np
import scipy.fftpack
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import scipy.io as sio
from collections import Counter
from sklearn import preprocessing
from scipy import stats
import numpy.polynomial.polynomial as poly
# import plotly 
pd.set_option('display.max_rows', 500)

from sklearn import preprocessing
from sklearn import svm, neighbors, metrics, cross_validation, preprocessing
from sklearn.externals import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, silhouette_score
from sklearn.cluster import KMeans, DBSCAN
from scipy import *
from scipy.stats import *
from scipy.signal import *
from collections import Counter
from sklearn.metrics import *
from sklearn.metrics import precision_recall_fscore_support as score




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

    Author
    ------
    Shibo(shibozhang2015@u.northwestern.edu)
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



def underSample(X,k):
    print ("Undersampling ...")

    # remove Others sample randomly to balance the training set
    label = X['label'].as_matrix()

    Nothers = np.count_nonzero(label == 0)
    Nfeeding = np.count_nonzero(label == 1)

    print ("Nothers " + str(Nothers) + " Nfeeding: " +  str(Nfeeding))

    if Nothers < Nfeeding * 5:
        return X
    else:
        others = np.where(label == 0)[0]
        feeding = np.where(label == 1)[0]
        import random 
        randIndex = random.sample(range(others.size), Nfeeding*k)
        others_reduced = others[randIndex]

        undersampled = list(others_reduced.ravel()) + list(feeding.ravel())
        undersampled.sort()

        return X.iloc[undersampled]



def balanced_subsample(x,y,subsample_size=1.0):

    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        elems = x[(y == yi)]
        class_xs.append((yi, elems))
        if min_elems == None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]

    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems*subsample_size)

    xs = []
    ys = []

    for ci,this_xs in class_xs:
        if len(this_xs) > use_elems:
            # np.random.shuffle(this_xs)
            this_xs = this_xs.reindex(np.random.permutation(this_xs.index))

        x_ = this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)

        xs.append(x_)
        ys.append(y_)

    # xs = np.concatenate(xs)
    # ys = np.concatenate(ys)
    xs = pd.concat(xs)
    ys = pd.Series(data=np.concatenate(ys),name='target')

    return xs,ys


def balanced_npsubsample(x,y,subsample_size=1.0):

    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        elems = x[(y == yi)]
        class_xs.append((yi, elems))
        if min_elems == None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]

    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems*subsample_size)

    xs = []
    ys = []

    for ci,this_xs in class_xs:
        if len(this_xs) > use_elems:
            np.random.shuffle(this_xs)
            # this_xs = this_xs.reindex(np.random.permutation(this_xs.index))

        x_ = this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)

        xs.append(x_)
        ys.append(y_)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    # xs = pd.concat(xs)
    # ys = pd.Series(data=np.concatenate(ys),name='target')

    return xs,ys



# input para: input_data , intervals_of_interest , timeString 
def markClassPeriod( df , col, intervals ):
    df[col] = 0
    for ts in intervals:
        a = str(ts[0])
        b = str(ts[1])
        df[col][(df.index >= a) & (df.index < b)] = 1
    return df

#  this markExistingClassPeriod() can totally substitute markClassPeriod()
def markExistingClassPeriod( df , col, label, intervals ):
    if col not in df.columns:
        df[col] = 0
    for ts in intervals:
        a = str(ts[0])
        b = str(ts[1])
        df[col][(df.index >= a) & (df.index < b)] = label
    return df

# the same as markExistingClassPeriod(), better name 

def mark( df , col, label, intervals ):
    if col not in df.columns:
        df[col] = 0
    for ts in intervals:
        a = str(ts[0])
        b = str(ts[1])
        df[col][(df.index >= a) & (df.index < b)] = label
    return df


def extractFeatures(input_data , intervals , window_size , start_time , threshold , labeled_outfile) :

    data_label = pd.DataFrame()
    data_label = markClassPeriod( input_data , intervals , timeString )

#********************   save to file   ********************
    if save_flg:
        data_label.to_csv(dRF + labeled_outfile)
  
    data_labelwin = extractWindows(data_label , window_size , start_time , threshold , timeString)
    
    featDF = pd.DataFrame(data_labelwin[1:] , columns=data_labelwin[0])
    return featDF

def poly_fit(Series,order):
    accx = Series.as_matrix()
    x = range(len(accx))
    x = np.linspace(x[0]/31, x[-1]/31, num=len(x))
    coefs = poly.polyfit(x, accx, 5)
#     x_new = range(len(x))
#     x_new = np.linspace(x_new[0], x_new[-1], num=len(x_new))
#     ffit = poly.polyval(x_new, coefs)
#     plt.plot(x_new, ffit)
    
    return coefs

def show_fft(y, freq):
    # Number of samplepoints
    N = y.shape[0]
    yf = scipy.fftpack.fft(y)
    amp = 2.0/N * np.abs(yf[:N/2])
    return amp[1:]


def cal_fft(y, freq):
    # Number of samplepoints
    N = y.shape[0]
    yf = scipy.fftpack.fft(y)
    amp = 2.0/N * np.abs(yf[:int(N/2)])
    return sum(i*i for i in amp)


def cal_energy_all(y, freq):
    # Number of samplepoints
    N = y.shape[0]
    yf = scipy.fftpack.fft(y)
    amp = 2.0/N * np.abs(yf[:int(N/2)])
    return sum(i*i for i in amp)


# return fft except the foundamental frequency component
def cal_energy_wo_bf(y, freq):
    # Number of samplepoints
    N = y.shape[0]
    yf = scipy.fftpack.fft(y)
    amp = 2.0/N * np.abs(yf[:int(N/2)])
    return sum(i*i for i in amp[1:])

# return the foundamental/basic frequency component
def cal_energy_bf(y, freq):
    # Number of samplepoints
    N = y.shape[0]
    yf = scipy.fftpack.fft(y)
    amp = 2.0/N * np.abs(yf[:int(N/2)])
    return sum(amp[0]*amp[0])


def delta_pitch_or_roll(a):
    diff = [t - s for s, t in zip(a, a[1:])]
    delta_pos_a = sum(i for i in diff if i > 0)
    delta_neg_a = abs(sum(i for i in diff if i < 0))

    return (delta_pos_a+delta_neg_a)

def iter_filter(a):
    b = [a[0]]
    for t in range(len(a)-1):
        b.append(0.3*a[t] + 0.7*b[t])
    return b

def df_iter_flt_norm(df):
    Linear_Accel_x = df['Linear_Accel_x'].as_matrix()
    Linear_Accel_y = df['Linear_Accel_y'].as_matrix()
    Linear_Accel_z = df['Linear_Accel_z'].as_matrix()
    Angular_Velocity_x = df['Angular_Velocity_x'].as_matrix()
    Angular_Velocity_y = df['Angular_Velocity_y'].as_matrix()
    Angular_Velocity_z = df['Angular_Velocity_z'].as_matrix()

    Linear_Accel_x = iter_filter(Linear_Accel_x)
    Linear_Accel_y = iter_filter(Linear_Accel_y)
    Linear_Accel_z = iter_filter(Linear_Accel_z)
    Angular_Velocity_x = iter_filter(Angular_Velocity_x)
    Angular_Velocity_y = iter_filter(Angular_Velocity_y)
    Angular_Velocity_z = iter_filter(Angular_Velocity_z)

    Linear_Accel_x = stats.zscore(Linear_Accel_x)
    Linear_Accel_y = stats.zscore(Linear_Accel_y)
    Linear_Accel_z = stats.zscore(Linear_Accel_z)
    Angular_Velocity_x = stats.zscore(Angular_Velocity_x)
    Angular_Velocity_y = stats.zscore(Angular_Velocity_y)
    Angular_Velocity_z = stats.zscore(Angular_Velocity_z)

    df['Linear_Accel_x'] = Linear_Accel_x
    df['Linear_Accel_y'] = Linear_Accel_y
    df['Linear_Accel_z'] = Linear_Accel_z
    df['Angular_Velocity_x'] = Angular_Velocity_x
    df['Angular_Velocity_y'] = Angular_Velocity_y
    df['Angular_Velocity_z'] = Angular_Velocity_z
    return df

def filter(df):
    flt_para = 10
    df.Linear_Accel_x = pd.rolling_mean(df.Linear_Accel_x, flt_para)
    df.Linear_Accel_y = pd.rolling_mean(df.Linear_Accel_y, flt_para)
    df.Linear_Accel_z = pd.rolling_mean(df.Linear_Accel_z, flt_para)
    
    df.Angular_Velocity_x = pd.rolling_mean(df.Angular_Velocity_x, flt_para)
    df.Angular_Velocity_y = pd.rolling_mean(df.Angular_Velocity_y, flt_para)
    df.Angular_Velocity_z = pd.rolling_mean(df.Angular_Velocity_z, flt_para)
    
    df.pitch_deg = pd.rolling_mean(df.pitch_deg, flt_para)
    df.roll_deg = pd.rolling_mean(df.roll_deg, flt_para)
    
    df = df.dropna()
    return df


def add_pitch_roll(df):
    Linear_Accel_x = df['Linear_Accel_x'].as_matrix()
    Linear_Accel_y = df['Linear_Accel_y'].as_matrix()
    Linear_Accel_z = df['Linear_Accel_z'].as_matrix()
    
    # strategy 1:
    pitch_deg = 180 * np.arctan (Linear_Accel_y/np.sqrt(Linear_Accel_x*Linear_Accel_x + Linear_Accel_z*Linear_Accel_z))/pi
    roll_deg = 180 * np.arctan (-Linear_Accel_x/Linear_Accel_z)/pi
    
    # strategy 2:
#     pitch_deg = 180 * np.arctan (Linear_Accel_x/np.sqrt(Linear_Accel_y*Linear_Accel_y + Linear_Accel_z*Linear_Accel_z))/pi;
#     roll_deg = 180 * np.arctan (Linear_Accel_y/np.sqrt(Linear_Accel_x*Linear_Accel_x + Linear_Accel_z*Linear_Accel_z))/pi;
#     yaw_deg = 180 * np.arctan (Linear_Accel_z/np.sqrt(Linear_Accel_x*Linear_Accel_x + Linear_Accel_z*Linear_Accel_z))/pi;
    
    df["pitch_deg"] = pitch_deg
    df["roll_deg"] = roll_deg
    return df

def gen_feat(df):
    
    Linear_Accel_x = df['Linear_Accel_x'].as_matrix()
    Linear_Accel_y = df['Linear_Accel_y'].as_matrix()
    Linear_Accel_z = df['Linear_Accel_z'].as_matrix()
    Angular_Velocity_x = df['Angular_Velocity_x'].as_matrix()
    Angular_Velocity_y = df['Angular_Velocity_y'].as_matrix()
    Angular_Velocity_z = df['Angular_Velocity_z'].as_matrix()
#     X_scaled = preprocessing.scale(X)
    
    featL = ["mean","median","max","min","skew","RMS","kurtosis","qurt1",'quart3','irq','stdev']#,'5th_coef','4th_coef','3rd_coef','2nd_coef','1st_coef'
    headlist = list(df.keys())
    header = []
    
    for key in headlist:
        for feat in featL:
            one = key + "_" + feat
            header.extend([one])
            
    header.extend(["cov_acc_x_y","cov_acc_y_z","cov_acc_x_z", 
                   "cov_gyro_x_y","cov_gyro_y_z","cov_gyro_x_z", 
                   "cov_acc_gyro_x","cov_acc_gyro_y","cov_acc_gyro_z",
                   "cov_acc_x_gyro_y","cov_acc_x_gyro_z",
                   "cov_acc_y_gyro_x","cov_acc_y_gyro_z",
                   "cov_acc_z_gyro_x","cov_acc_z_gyro_y",
                   "energy_accx","energy_accy","energy_accz", "energy_acc_xyz",
                   "delta_pitch","delta_roll","duration"])
#     
    allfeats = []
    allfeats.append(header)
    
    keylist = list(df.keys())
    # print(keylist)
    if 1:
        features = []
#         print(keylist)
        for key in keylist:
            win = df[key]
            irq = win.quantile(q=0.75) - win.quantile(q=0.25)
            features.extend([float(win.mean())]) #mean
            features.extend([float(win.median())]) #median
            features.extend([float(win.max())]) #max
            features.extend([float(win.min())]) #min
            features.extend([float(win.skew())]) #skew
            features.extend([float(sqrt((win**2).mean()))]) #RMS
            features.extend([float(win.kurt())]) #kurtosis
            features.extend([float(win.quantile(q=0.25))]) #1st quartile
            features.extend([float(win.quantile(q=0.75))]) #3rd quartile
            features.extend([float(irq)])#inter quartile range
            #features.extend([float(win.corr())]) #correlation
            features.extend([float(win.std())])#std dev
            # coef_list = poly_fit(win,5)
            # features.extend([float(coef_list[-1])])
            # features.extend([float(coef_list[-2])])
            # features.extend([float(coef_list[-3])])
            # features.extend([float(coef_list[-4])])
            # features.extend([float(coef_list[-5])])
            

        features.extend([float(df["Linear_Accel_x"].cov(df["Linear_Accel_y"]))])
        features.extend([float(df["Linear_Accel_y"].cov(df["Linear_Accel_z"]))])
        features.extend([float(df["Linear_Accel_x"].cov(df["Linear_Accel_z"]))])
        features.extend([float(df["Angular_Velocity_x"].cov(df["Angular_Velocity_y"]))])
        features.extend([float(df["Angular_Velocity_y"].cov(df["Angular_Velocity_z"]))])
        features.extend([float(df["Angular_Velocity_x"].cov(df["Angular_Velocity_z"]))])

        
        features.extend([float(df["Linear_Accel_x"].cov(df["Angular_Velocity_x"]))])
        features.extend([float(df["Linear_Accel_y"].cov(df["Angular_Velocity_y"]))])
        features.extend([float(df["Linear_Accel_z"].cov(df["Angular_Velocity_z"]))])
        
        features.extend([float(df["Linear_Accel_x"].cov(df["Angular_Velocity_y"]))])
        features.extend([float(df["Linear_Accel_x"].cov(df["Angular_Velocity_z"]))])
        features.extend([float(df["Linear_Accel_y"].cov(df["Angular_Velocity_x"]))])
        features.extend([float(df["Linear_Accel_y"].cov(df["Angular_Velocity_z"]))])
        features.extend([float(df["Linear_Accel_z"].cov(df["Angular_Velocity_x"]))])
        features.extend([float(df["Linear_Accel_z"].cov(df["Angular_Velocity_y"]))])
        
#         energy feature for 3 axes and total
        fftx = cal_energy_all(Linear_Accel_x, 31)
        features.extend([float(fftx)])
        ffty = cal_energy_all(Linear_Accel_y, 31)
        features.extend([float(ffty)])
        fftz = cal_energy_all(Linear_Accel_z, 31)
        features.extend([float(fftz)])
        features.extend([fftx*fftx + ffty*ffty + fftz*fftz])
#         delta_pitch and delta_roll
        pitch_deg_arr = df['pitch_deg'].as_matrix().tolist()
        features.extend([delta_pitch_or_roll(pitch_deg_arr)])
        roll_deg_arr = df['roll_deg'].as_matrix().tolist()
        features.extend([delta_pitch_or_roll(roll_deg_arr)])
#         duration
        features.extend([df.shape[0]])
        allfeats.append(features)
        
    return allfeats



def gen_feat_reduced(df):
    
    Linear_Accel_x = df['Linear_Accel_x'].as_matrix()
    Linear_Accel_y = df['Linear_Accel_y'].as_matrix()
    Linear_Accel_z = df['Linear_Accel_z'].as_matrix()
    Angular_Velocity_x = df['Angular_Velocity_x'].as_matrix()
    Angular_Velocity_y = df['Angular_Velocity_y'].as_matrix()
    Angular_Velocity_z = df['Angular_Velocity_z'].as_matrix()
    
    featL = ["mean","median","max","min","skew","RMS","kurtosis","qurt1",'quart3','irq','stdev']#,'5th_coef','4th_coef','3rd_coef','2nd_coef','1st_coef'
    headlist = list(df.keys())
    header = []
    
    for key in headlist:
        for feat in featL:
            one = key + "_" + feat
            header.extend([one])
            
    header.extend(["duration"])
#     
    allfeats = []
    allfeats.append(header)
    
    keylist = list(df.keys())
    # print(keylist)
    if 1:
        features = []
#         print(keylist)
        for key in keylist:
            win = df[key]
            irq = win.quantile(q=0.75) - win.quantile(q=0.25)
            
            features.extend([float(win.mean())]) #mean
            features.extend([float(win.median())]) #median
            features.extend([float(win.max())]) #max
            features.extend([float(win.min())]) #min
            features.extend([float(win.skew())]) #skew
            features.extend([float(sqrt((win**2).mean()))]) #RMS
            features.extend([float(win.kurt())]) #kurtosis
            features.extend([float(win.quantile(q=0.25))]) #1st quartile
            features.extend([float(win.quantile(q=0.75))]) #3rd quartile
            features.extend([float(irq)])#inter quartile range
            #features.extend([float(win.corr())]) #correlation
            features.extend([float(win.std())])#std dev
            
        features.extend([df.shape[0]])
        allfeats.append(features)
        
    return allfeats




def gen_acc_fftsq_energy(df, freq):
    
    Linear_Accel_x = df['Linear_Accel_x'].as_matrix()
    Linear_Accel_y = df['Linear_Accel_y'].as_matrix()
    Linear_Accel_z = df['Linear_Accel_z'].as_matrix()
    Angular_Velocity_x = df['Angular_Velocity_x'].as_matrix()
    Angular_Velocity_y = df['Angular_Velocity_y'].as_matrix()
    Angular_Velocity_z = df['Angular_Velocity_z'].as_matrix()
#     X_scaled = preprocessing.scale(X)

    header = []
    header.extend(["energy_acc_xxyyzz"])
#     
    allfeats = []
    allfeats.append(header)
    
    keylist = list(df.keys())
    # print(keylist)
    if 1:
        features = []

        fftx = cal_fft(Linear_Accel_x, freq)
        ffty = cal_fft(Linear_Accel_y, freq)
        fftz = cal_fft(Linear_Accel_z, freq)
        features.extend([fftx*fftx + ffty*ffty + fftz*fftz])

        allfeats.append(features)
        
    return allfeats



def gen_energy(df, freq):
    
    Linear_Accel_x = df['Linear_Accel_x'].as_matrix()
    Linear_Accel_y = df['Linear_Accel_y'].as_matrix()
    Linear_Accel_z = df['Linear_Accel_z'].as_matrix()
    Angular_Velocity_x = df['Angular_Velocity_x'].as_matrix()
    Angular_Velocity_y = df['Angular_Velocity_y'].as_matrix()
    Angular_Velocity_z = df['Angular_Velocity_z'].as_matrix()
#     X_scaled = preprocessing.scale(X)

    header = []
    header.extend([ "energy_acc_xyz","orientation_acc_xyz","energy_orientation","energy_acc_xxyyzz",
        "energy_ang_xyz", "energy_ang_xyz_regularized"])
#     
    allfeats = []
    allfeats.append(header)
    
    keylist = list(df.keys())
    # print(keylist)
    if 1:
        features = []
#           energy feature for 3 axes and total
#           energy_acc_xyz
        fftx = cal_energy_wo_bf(Linear_Accel_x, freq)
        ffty = cal_energy_wo_bf(Linear_Accel_y, freq)
        fftz = cal_energy_wo_bf(Linear_Accel_z, freq)
        features.extend([fftx + ffty + fftz])

#           orientation_acc_xyz
        orix = cal_energy_bf(Linear_Accel_x, freq)
        oriy = cal_energy_bf(Linear_Accel_y, freq)
        oriz = cal_energy_bf(Linear_Accel_z, freq)
        features.extend([orix + oriy + oriz])

#           energy + orientation_acc_xyz
        ener_orix = fftx + orix
        ener_oriy = ffty + oriy
        ener_oriz = fftz + oriz
        features.extend([ener_orix + ener_oriy + ener_oriz])


        fftx = cal_fft(Linear_Accel_x, freq)
        ffty = cal_fft(Linear_Accel_y, freq)
        fftz = cal_fft(Linear_Accel_z, freq)
        features.extend([fftx*fftx + ffty*ffty + fftz*fftz])

        # energy_ang_xxyyzz
        fftx = cal_fft(Angular_Velocity_x, freq)
        ffty = cal_fft(Angular_Velocity_y, freq)
        fftz = cal_fft(Angular_Velocity_z, freq)
        features.extend([fftx + ffty + fftz])

        # energy_ang_xxyyzz
        fftx = cal_fft(Angular_Velocity_x/360, freq)
        ffty = cal_fft(Angular_Velocity_y/360, freq)
        fftz = cal_fft(Angular_Velocity_z/360, freq)
        features.extend([fftx + ffty + fftz])

        allfeats.append(features)
        
    return allfeats



def plot_acc(r_df, starttime, endtime, title):
    
    mask = (r_df['Time'] > starttime) & (r_df['Time'] <= endtime)
    r_df = r_df.loc[mask]

    r_df_accel = r_df[[ 'Linear_Accel_x','Linear_Accel_y','Linear_Accel_z']]
    f = plt.figure(figsize=(15,15))
    styles1 = ['b-','r-','y-']
    r_df_accel.plot(style=styles1,ax=f.gca())
    plt.title(title, color='black')
    return r_df_accel

def plot_gyro(r_df, starttime, endtime, title):
#     mask = (r_df['Time'] > '2016-08-09 01:20:00') & (r_df['Time'] <= '2016-08-09 01:23:10')
    mask = (r_df['Time'] > starttime) & (r_df['Time'] <= endtime)
    r_df = r_df.loc[mask]
    
    r_df_gyro = r_df[[ 'Angular_Velocity_x','Angular_Velocity_y','Angular_Velocity_z']]
    f = plt.figure(figsize=(15,15))
    styles1 = ['b-','r-','y-']
    r_df_gyro.plot(style=styles1,ax=f.gca())
#     plt.title('answer phone call right hand gyro', color='black')
    plt.title(title, color='black')
    return r_df_gyro





def clf_cm_pickle_rcall(classifier, X_test, y_test):#

    # Run classifier
    y_pred = classifier.predict(X_test)    
    
    # ct = pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True).apply(lambda r: r/r.sum(), axis=1)
    ct = pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
    print(ct)
    # ct.to_csv(cm_file)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    accuracy = sum(cm[i,i] for i in range(len(set(y_test))))/sum(sum(cm[i] for i in range(len(set(y_test)))))
    recall_all = sum(cm[i,i]/sum(cm[i,j] for j in range(len(set(y_test)))) for i in range(len(set(y_test))))/(len(set(y_test)))
    precision_all = sum(cm[i,i]/sum(cm[j,i] for j in range(len(set(y_test)))) for i in range(len(set(y_test))))/(len(set(y_test)))
    fscore_all = sum(2*(cm[i,i]/sum(cm[i,j] for j in range(len(set(y_test)))))*(cm[i,i]/sum(cm[j,i] for j in range(len(set(y_test)))))/(cm[i,i]/sum(cm[i,j] for j in range(len(set(y_test))))+cm[i,i]/sum(cm[j,i] for j in range(len(set(y_test))))) for i in range(len(set(y_test))))/len(set(y_test))
    
    # this part can be improved
    if cm.shape == (2,2):
        TP = cm[1,1]
        FP = cm[0,1]
        TN = cm[0,0]
        FN = cm[1,0]
    else:
        TN = cm[0,0]
        FN = cm[1,0]
        TP = 0
        FP = 0


    # Precision for Positive = TP/(TP + FP)
    if TP + FP == 0:
        prec_pos = float('nan')
    else:
        prec_pos = TP/(TP + FP)

    if TP + FN == 0:
        recall_pos = float('nan')
    else:
        recall_pos = TP/(TP + FN)

    # F1 score for positive = 2 * precision * recall / (precision + recall)….or it can be F1= 2*TP/(2*TP + FP+ FN)
    f1_pos = 2*TP/(TP*2 + FP+ FN)
    print(f1_pos)
    # TPR = TP/(TP+FN)

    if TP + FN == 0:
        TPR = float('nan')
    else:
        TPR = cm[1,1]/sum(cm[1,j] for j in range(len(set(y_test))))

    # FPR = FP/(FP+TN)
    if FP + TN == 0:
        FPR = float('nan')
    else:
        FPR = cm[0,1]/sum(cm[0,j] for j in range(len(set(y_test))))
    # specificity = TN/(FP+TN)
    if FP+TN == 0:
        Specificity = float('nan')
    else:
        Specificity = cm[0,0]/sum(cm[0,j] for j in range(len(set(y_test))))

    MCC = matthews_corrcoef(y_test, y_pred)

    CKappa = metrics.cohen_kappa_score(y_test, y_pred)

    # w_acc = (TP*20 + TN)/ [(TP+FN)*20 + (TN+FP)] if 20:1 ratio of non-feeding to feeding

    if TP+FN == 0:
        ratio = float('nan')
    else:
        ratio = (TN+FP)/(TP+FN)

    w_acc = (TP*ratio + TN)/ ((TP+FN)*ratio + (TN+FP))

    # Show confusion matrix in a separate window
#     plt.matshow(cm)
#     plt.title('Confusion matrix')
#     plt.colorbar()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.show()
    
#     print(accuracy, recall_all, precision_all, fscore_all)
    return prec_pos, recall_pos, f1_pos, TPR, FPR, Specificity, MCC, CKappa, w_acc, cm, y_pred




def clf_cm_pickle(classifier, X_test, y_test):#

    # Run classifier
    y_pred = classifier.predict(X_test)    
    
    # ct = pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True).apply(lambda r: r/r.sum(), axis=1)
    ct = pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
    print(ct)
    # ct.to_csv(cm_file)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    accuracy = sum(cm[i,i] for i in range(len(set(y_test))))/sum(sum(cm[i] for i in range(len(set(y_test)))))
    recall_all = sum(cm[i,i]/sum(cm[i,j] for j in range(len(set(y_test)))) for i in range(len(set(y_test))))/(len(set(y_test)))
    precision_all = sum(cm[i,i]/sum(cm[j,i] for j in range(len(set(y_test)))) for i in range(len(set(y_test))))/(len(set(y_test)))
    fscore_all = sum(2*(cm[i,i]/sum(cm[i,j] for j in range(len(set(y_test)))))*(cm[i,i]/sum(cm[j,i] for j in range(len(set(y_test)))))/(cm[i,i]/sum(cm[i,j] for j in range(len(set(y_test))))+cm[i,i]/sum(cm[j,i] for j in range(len(set(y_test))))) for i in range(len(set(y_test))))/len(set(y_test))
    
    # this part can be improved
    if cm.shape == (2,2):
        TP = cm[1,1]
        FP = cm[0,1]
        TN = cm[0,0]
        FN = cm[1,0]
    else:
        TN = cm[0,0]
        FN = cm[1,0]
        TP = 0
        FP = 0


    # Precision for Positive = TP/(TP + FP)
    if TP + FP == 0:
        prec_pos = float('nan')
    else:
        prec_pos = TP/(TP + FP)

    # F1 score for positive = 2 * precision * recall / (precision + recall)….or it can be F1= 2*TP/(2*TP + FP+ FN)
    f1_pos = 2*TP/(TP*2 + FP+ FN)
    print(f1_pos)
    # TPR = TP/(TP+FN)

    if TP + FN == 0:
        TPR = float('nan')
    else:
        TPR = cm[1,1]/sum(cm[1,j] for j in range(len(set(y_test))))

    # FPR = FP/(FP+TN)
    if FP + TN == 0:
        FPR = float('nan')
    else:
        FPR = cm[0,1]/sum(cm[0,j] for j in range(len(set(y_test))))
    # specificity = TN/(FP+TN)
    if FP+TN == 0:
        Specificity = float('nan')
    else:
        Specificity = cm[0,0]/sum(cm[0,j] for j in range(len(set(y_test))))

    MCC = matthews_corrcoef(y_test, y_pred)

    CKappa = metrics.cohen_kappa_score(y_test, y_pred)

    # w_acc = (TP*20 + TN)/ [(TP+FN)*20 + (TN+FP)] if 20:1 ratio of non-feeding to feeding

    if TP+FN == 0:
        ratio = float('nan')
    else:
        ratio = (TN+FP)/(TP+FN)

    w_acc = (TP*ratio + TN)/ ((TP+FN)*ratio + (TN+FP))

    # Show confusion matrix in a separate window
#     plt.matshow(cm)
#     plt.title('Confusion matrix')
#     plt.colorbar()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.show()
    
#     print(accuracy, recall_all, precision_all, fscore_all)
    return prec_pos, f1_pos, TPR, FPR, Specificity, MCC, CKappa, w_acc, cm, y_pred




def clf_cm(X_train, X_test, y_train, y_test):#

    # Run classifier
    classifier = RandomForestClassifier(n_estimators=185)
    y_pred = classifier.fit(X_train, y_train).predict(X_test)    
    
    # ct = pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True).apply(lambda r: r/r.sum(), axis=1)
    ct = pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
    # print(ct)
    # ct.to_csv(cm_file)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    accuracy = sum(cm[i,i] for i in range(len(set(y_train))))/sum(sum(cm[i] for i in range(len(set(y_train)))))
    recall_all = sum(cm[i,i]/sum(cm[i,j] for j in range(len(set(y_train)))) for i in range(len(set(y_train))))/(len(set(y_train)))
    precision_all = sum(cm[i,i]/sum(cm[j,i] for j in range(len(set(y_train)))) for i in range(len(set(y_train))))/(len(set(y_train)))
    fscore_all = sum(2*(cm[i,i]/sum(cm[i,j] for j in range(len(set(y_train)))))*(cm[i,i]/sum(cm[j,i] for j in range(len(set(y_train)))))/(cm[i,i]/sum(cm[i,j] for j in range(len(set(y_train))))+cm[i,i]/sum(cm[j,i] for j in range(len(set(y_train))))) for i in range(len(set(y_train))))/len(set(y_train))
    
    TP = cm[1,1]
    FP = cm[0,1]
    TN = cm[0,0]
    FN = cm[1,0]
    # Precision for Positive = TP/(TP + FP)
    prec_pos = TP/(TP + FP)
    # F1 score for positive = 2 * precision * recall / (precision + recall)….or it can be F1= 2*TP/(2*TP + FP+ FN)
    f1_pos = 2*TP/(TP*2 + FP+ FN)
    # TPR = TP/(TP+FN)
    TPR = cm[1,1]/sum(cm[1,j] for j in range(len(set(y_train))))
    # FPR = FP/(FP+TN)
    FPR = cm[0,1]/sum(cm[0,j] for j in range(len(set(y_train))))
    # specificity = TN/(FP+TN)
    Specificity = cm[0,0]/sum(cm[0,j] for j in range(len(set(y_train))))

    MCC = matthews_corrcoef(y_test, y_pred)

    CKappa = metrics.cohen_kappa_score(y_test, y_pred)

    # w_acc = (TP*20 + TN)/ [(TP+FN)*20 + (TN+FP)] if 20:1 ratio of non-feeding to feeding
    ratio = (TN+FP)/(TP+FN)

    w_acc = (TP*ratio + TN)/ ((TP+FN)*ratio + (TN+FP))

    # Show confusion matrix in a separate window
#     plt.matshow(cm)
#     plt.title('Confusion matrix')
#     plt.colorbar()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.show()
    
#     print(accuracy, recall_all, precision_all, fscore_all)
    return prec_pos, f1_pos, TPR, FPR, Specificity, MCC, CKappa, w_acc, cm


def calc_cm_rcall(y_test, y_pred):#

    
    # ct = pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True).apply(lambda r: r/r.sum(), axis=1)
    ct = pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
    # print(ct)
    # ct.to_csv(cm_file)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    accuracy = sum(cm[i,i] for i in range(len(set(y_test))))/sum(sum(cm[i] for i in range(len(set(y_test)))))
    recall_all = sum(cm[i,i]/sum(cm[i,j] for j in range(len(set(y_test)))) for i in range(len(set(y_test))))/(len(set(y_test)))
    precision_all = sum(cm[i,i]/sum(cm[j,i] for j in range(len(set(y_test)))) for i in range(len(set(y_test))))/(len(set(y_test)))
    fscore_all = sum(2*(cm[i,i]/sum(cm[i,j] for j in range(len(set(y_test)))))*(cm[i,i]/sum(cm[j,i] for j in range(len(set(y_test)))))/(cm[i,i]/sum(cm[i,j] for j in range(len(set(y_test))))+cm[i,i]/sum(cm[j,i] for j in range(len(set(y_test))))) for i in range(len(set(y_test))))/len(set(y_test))
    
    TP = cm[1,1]
    FP = cm[0,1]
    TN = cm[0,0]
    FN = cm[1,0]
    # Precision for Positive = TP/(TP + FP)
    prec_pos = TP/(TP + FP)

    recall_pos = TP/(TP+FN)

    # F1 score for positive = 2 * precision * recall / (precision + recall)….or it can be F1= 2*TP/(2*TP + FP+ FN)
    f1_pos = 2*TP/(TP*2 + FP+ FN)
    # TPR = TP/(TP+FN)
    TPR = cm[1,1]/sum(cm[1,j] for j in range(len(set(y_test))))
    # FPR = FP/(FP+TN)
    FPR = cm[0,1]/sum(cm[0,j] for j in range(len(set(y_test))))
    # specificity = TN/(FP+TN)
    Specificity = cm[0,0]/sum(cm[0,j] for j in range(len(set(y_test))))

    MCC = matthews_corrcoef(y_test, y_pred)

    CKappa = metrics.cohen_kappa_score(y_test, y_pred)

    # w_acc = (TP*20 + TN)/ [(TP+FN)*20 + (TN+FP)] if 20:1 ratio of non-feeding to feeding
    ratio = (TN+FP)/(TP+FN)

    w_acc = (TP*ratio + TN)/ ((TP+FN)*ratio + (TN+FP))

    # Show confusion matrix in a separate window
#     plt.matshow(cm)
#     plt.title('Confusion matrix')
#     plt.colorbar()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.show()
    
#     print(accuracy, recall_all, precision_all, fscore_all)
    return prec_pos, recall_pos, f1_pos, TPR, FPR, Specificity, MCC, CKappa, w_acc, cm




def calc_cm(y_test, y_pred):#

    
    # ct = pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True).apply(lambda r: r/r.sum(), axis=1)
    ct = pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
    print(ct)
    # ct.to_csv(cm_file)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    accuracy = sum(cm[i,i] for i in range(len(set(y_test))))/sum(sum(cm[i] for i in range(len(set(y_test)))))
    recall_all = sum(cm[i,i]/sum(cm[i,j] for j in range(len(set(y_test)))) for i in range(len(set(y_test))))/(len(set(y_test)))
    precision_all = sum(cm[i,i]/sum(cm[j,i] for j in range(len(set(y_test)))) for i in range(len(set(y_test))))/(len(set(y_test)))
    fscore_all = sum(2*(cm[i,i]/sum(cm[i,j] for j in range(len(set(y_test)))))*(cm[i,i]/sum(cm[j,i] for j in range(len(set(y_test)))))/(cm[i,i]/sum(cm[i,j] for j in range(len(set(y_test))))+cm[i,i]/sum(cm[j,i] for j in range(len(set(y_test))))) for i in range(len(set(y_test))))/len(set(y_test))
    
    TP = cm[1,1]
    FP = cm[0,1]
    TN = cm[0,0]
    FN = cm[1,0]
    # Precision for Positive = TP/(TP + FP)
    prec_pos = TP/(TP + FP)
    # F1 score for positive = 2 * precision * recall / (precision + recall)….or it can be F1= 2*TP/(2*TP + FP+ FN)
    f1_pos = 2*TP/(TP*2 + FP+ FN)
    # TPR = TP/(TP+FN)
    TPR = cm[1,1]/sum(cm[1,j] for j in range(len(set(y_test))))
    # FPR = FP/(FP+TN)
    FPR = cm[0,1]/sum(cm[0,j] for j in range(len(set(y_test))))
    # specificity = TN/(FP+TN)
    Specificity = cm[0,0]/sum(cm[0,j] for j in range(len(set(y_test))))

    MCC = matthews_corrcoef(y_test, y_pred)

    CKappa = metrics.cohen_kappa_score(y_test, y_pred)

    # w_acc = (TP*20 + TN)/ [(TP+FN)*20 + (TN+FP)] if 20:1 ratio of non-feeding to feeding
    ratio = (TN+FP)/(TP+FN)

    w_acc = (TP*ratio + TN)/ ((TP+FN)*ratio + (TN+FP))

    # Show confusion matrix in a separate window
#     plt.matshow(cm)
#     plt.title('Confusion matrix')
#     plt.colorbar()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.show()
    
#     print(accuracy, recall_all, precision_all, fscore_all)
    return prec_pos, f1_pos, TPR, FPR, Specificity, MCC, CKappa, w_acc, cm


def calc_multi_cm(y_test, y_pred):#

    
    # ct = pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True).apply(lambda r: r/r.sum(), axis=1)
    ct = pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
    print(ct)
    # ct.to_csv(cm_file)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    accuracy = sum(cm[i,i] for i in range(len(set(y_test))))/sum(sum(cm[i] for i in range(len(set(y_test)))))
    recall_all = sum(cm[i,i]/sum(cm[i,j] for j in range(len(set(y_test)))) for i in range(len(set(y_test))))/(len(set(y_test)))
    precision_all = sum(cm[i,i]/sum(cm[j,i] for j in range(len(set(y_test)))) for i in range(len(set(y_test))))/(len(set(y_test)))
    fscore_all = sum(2*(cm[i,i]/sum(cm[i,j] for j in range(len(set(y_test)))))*(cm[i,i]/sum(cm[j,i] for j in range(len(set(y_test)))))/(cm[i,i]/sum(cm[i,j] for j in range(len(set(y_test))))+cm[i,i]/sum(cm[j,i] for j in range(len(set(y_test))))) for i in range(len(set(y_test))))/len(set(y_test))

    # Show confusion matrix in a separate window
#     plt.matshow(cm)
#     plt.title('Confusion matrix')
#     plt.colorbar()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.show()
    
    return accuracy, recall_all, precision_all, fscore_all



def k_fold_split(XY, k, n):
    # k is the number of folds
    # n is the index of current fold, n>=0 , n<k
    length = len(XY)
    test_ind = np.arange(n, length, k)
    train_ind = [x for x in list(range(length)) if x not in test_ind]

    return XY[train_ind], XY[test_ind]



def tt_split(XY, train_ratio):
    # eg: train_ratio = 0.7
    length = len(XY)
    # print(length)
    test_enum = range(int((1-train_ratio)*10))
    test_ind = []
    for i in test_enum:
        test_ind = test_ind + list(range(i, length, 10))

    # test_ind = np.arange(n, length, k)
    train_ind = [x for x in list(range(length)) if x not in test_ind]

    return XY[train_ind], XY[test_ind]



def tt_split_df(XY, train_ratio):
    # eg: train_ratio = 0.7
    length = len(XY)
    # print(length)
    test_enum = range(int((1-train_ratio)*10))
    test_ind = []
    for i in test_enum:
        test_ind = test_ind + list(range(i, length, 10))

    # test_ind = np.arange(n, length, k)
    train_ind = [x for x in list(range(length)) if x not in test_ind]

    return XY.iloc[train_ind], XY.iloc[test_ind]




def tt_split_rand(XY, train_ratio, seed):
    # eg: train_ratio = 0.7
    import random

    numL = list(range(10))
    random.seed(seed)
    random.shuffle(numL)

    length = len(XY)
    test_enum = numL[0:int((1-train_ratio)*10)]
    test_ind = []
    for i in test_enum:
        test_ind = test_ind + list(range(i, length, 10))

    # test_ind = np.arange(n, length, k)
    train_ind = [x for x in list(range(length)) if x not in test_ind]

    return XY[train_ind], XY[test_ind]



def pointwise2headtail(pointwise_rpr):
    pw = pointwise_rpr
    diff = np.concatenate((pw[:],np.array([0]))) - np.concatenate((np.array([0]),pw[:]))
    ind_head = np.where(diff == 1)[0]
    ind_tail = np.where(diff == -1)[0]-1
    print(len(ind_tail))
    print(len(ind_head))

    headtail_rpr = np.vstack((ind_head, ind_tail)).T;

    return headtail_rpr














def read_r_df_test(subj, file, birthtime, deadtime):
    r_df = pd.read_csv(file)
    r_df = r_df[["Time","Angular_Velocity_x","Angular_Velocity_y","Angular_Velocity_z","Linear_Accel_x","Linear_Accel_y","Linear_Accel_z"]]
    r_df["unixtime"] = r_df["Time"]
    r_df["synctime"] = r_df["unixtime"]
    r_df['Time'] = pd.to_datetime(r_df['Time'],unit='ms',utc=True)
    r_df = r_df.set_index(['Time'])

    # to video absolute time
    r_df.index = r_df.index.tz_localize('UTC').tz_convert('US/Central')

    # cut and select the test part
    mask = ((r_df.index > birthtime) & (r_df.index < deadtime))
    r_df_test = r_df.loc[mask]
    
    return r_df_test

def read_r_df_test_st(subj, file, birthtime):
    r_df = pd.read_csv(file)
    r_df = r_df[["Time","Angular_Velocity_x","Angular_Velocity_y","Angular_Velocity_z","Linear_Accel_x","Linear_Accel_y","Linear_Accel_z"]]
    r_df["unixtime"] = r_df["Time"]
    r_df["synctime"] = r_df["unixtime"]
    r_df['Time'] = pd.to_datetime(r_df['Time'],unit='ms',utc=True)
    r_df = r_df.set_index(['Time'])

    # to video absolute time
    r_df.index = r_df.index.tz_localize('UTC').tz_convert('US/Central')

    # cut and select the test part
    mask = (r_df.index > birthtime)
    r_df_test = r_df.loc[mask]
    
    return r_df_test


def rm_black_out(r_df_test, annotDf):
    annot_blac = annotDf.loc[(annotDf["Annotation"]=="WeirdTimeJump")&(annotDf["MainCategory"]=="Confusing")]

    StartTime_list = list(annot_blac.StartTime.tolist())
    EndTime_list = list(annot_blac.EndTime.tolist())

    for n in range(len(StartTime_list)):
        r_df_test = r_df_test[(r_df_test.index < str(StartTime_list[n]))|(r_df_test.index > str(EndTime_list[n]))]

    return r_df_test


def importAnnoFile(annot_file):
    annotDf = pd.read_csv(annot_file, encoding = "ISO-8859-1")
    # print(annotDf['StartTime'])
    annotDf['StartTime'] = pd.to_datetime(annotDf['StartTime'],utc=True)
    annotDf['EndTime'] = pd.to_datetime(annotDf['EndTime'],utc=True)
    return annotDf

def processAnnot(annotDf):
    annotDf['drop'] = 0
    annotDf['duration'] = annotDf['EndTime'] - annotDf['StartTime']
    annotDf['drop'].loc[(annotDf['duration']<'00:00:01.5')&(annotDf["Annotation"]=="WeirdTimeJump")&(annotDf["MainCategory"]=="Confusing")]= 1
    annotDf = annotDf.loc[annotDf['drop'] == 0]

    annotDf  = annotDf[['StartTime','EndTime','Annotation','MainCategory']]
    return annotDf

def gen_energy_file(r_df_test, winsize, stride, freq, featsfile):
    i = 0
    allfeatDF = pd.DataFrame()

    while(i+winsize < r_df_test.shape[0]):    
        feat = gen_energy(r_df_test[i:i+winsize], freq)
        featDF = pd.DataFrame(feat[1:] , columns=feat[0])
        i += stride
        if i%100 == 0:
            print(i)
        allfeatDF = pd.concat([allfeatDF,featDF])

    return allfeatDF


def checkHandUpDown(annot_HU, annot_HD):
    for i in range(1,max(len(annot_HD),len(annot_HU))):
        if not (annot_HU.StartTime.iloc[i]>annot_HD.StartTime.iloc[i-1]) & (annot_HU.StartTime.iloc[i]<annot_HD.StartTime.iloc[i]):
            print("subj: "+subj)
            print("trouble line: "+str(i))
            print("annot_HandDown.StartTime of "+str(i-1)+" is "+str(annot_HD.StartTime.iloc[i-1]))
            print("annot_HandUp.StartTime of "+str(i)+" is "+str(annot_HU.StartTime.iloc[i]))
            print("annot_HandDown.StartTime of "+str(i)+" is "+str(annot_HD.StartTime.iloc[i]))
            return -1
    return 0



def unstrGenFeedingLabels(annotDf, r_df_test, activities):
    # firstly, extract the eating and drinking activites for 'Annotation' column
    # secondly, find HandUp and HandDown for 'MainCatetory' column from the dataframe by step 1


    # step 1:
    annot_f = pd.DataFrame()
    act_dur = []

    for i,activity in enumerate(activities):
        annot_f_tmp = annotDf.loc[(annotDf["Annotation"]==activity)]
        annot_f = pd.concat([annot_f, annot_f_tmp])

    annot_f = annot_f.sort_values(by='StartTime')

    # step 2:
    annot_HU = annot_f.loc[(annot_f["MainCategory"]=="HandUp")]
    annot_HD = annot_f.loc[(annot_f["MainCategory"]=="HandDown")]
    annot_HU = annot_HU.drop_duplicates()
    annot_HD = annot_HD.drop_duplicates()
    annot_HU.to_csv("../"+protocol+"/subject/"+subjfolder+"/f_HU.csv")
    annot_HD.to_csv("../"+protocol+"/subject/"+subjfolder+"/f_HD.csv")


    if len(annot_HU) != len(annot_HD):
        print("feeding gesture hand up and hand down in pairs")
        print(len(annot_HU))
        print(len(annot_HD))
        exit()

    if(checkHandUpDown(annot_HU, annot_HD)):
        print("feeding gesture error")
        print(len(annot_HU))
        print(len(annot_HD))
        exit()

    feeding_St_list = list(annot_HU.StartTime.tolist())
    feeding_Et_list = list(annot_HD.EndTime.tolist())
    dur = []
    for n in range(len(feeding_St_list)):
        dur.append([feeding_St_list[n],feeding_Et_list[n]])


    # step 3: label test data
    # mark( df , col, label, intervals ):
    r_df_test = mark(r_df_test, 'feedingClass', 1, dur)
    # print(act_dur)

    return r_df_test



def unstrGenNonfeedingLabels_tmp(annotDf, r_df_test, activities):
    # firstly, extract the eating and drinking activites for 'Annotation' column
    # secondly, find HandUp and HandDown for 'MainCatetory' column from the dataframe by step 1


    # step 1:
    annot_f = pd.DataFrame()
    act_dur = []

    for i,activity in enumerate(activities):
        annot_f_tmp = annotDf.loc[(annotDf["Annotation"]==activity)]
        annot_f = pd.concat([annot_f, annot_f_tmp])
    
    annot_f = annot_f.sort_values(by='StartTime')


    # step 2:
    annot_HU = annot_f.loc[(annot_f["MainCategory"]=="HandUp")]
    annot_HD = annot_f.loc[(annot_f["MainCategory"]=="HandDown")]
    annot_HU = annot_HU.drop_duplicates()
    annot_HD = annot_HD.drop_duplicates()

    annot_HU.to_csv("../"+protocol+"/subject/"+subjfolder+"/nf_HU.csv")
    annot_HD.to_csv("../"+protocol+"/subject/"+subjfolder+"/nf_HD.csv")


    if(checkHandUpDown(annot_HU, annot_HD)):
        print("non-feeding gesture error")
        print(len(annot_HU))
        print(len(annot_HD))
        exit()

    if len(annot_HU) != len(annot_HD):
        print("feeding gesture hand up and hand down not in pairs")
        print(len(annot_HU))
        print(len(annot_HD))
        exit()

    feeding_St_list = list(annot_HU.StartTime.tolist())
    feeding_Et_list = list(annot_HD.EndTime.tolist())
    dur = []
    for n in range(len(feeding_St_list)):
        dur.append([feeding_St_list[n],feeding_Et_list[n]])


    # step 3: label test data
    # mark( df , col, label, intervals ):
    r_df_test = mark(r_df_test, 'nonfeedingClass', 1, dur)
    # print(act_dur)

    return r_df_test



def genWinHandUpHoldingDownLabels(annotDf, r_df_test, activities):
    annot_HU = annotDf.loc[(annotDf["MainCategory"]=="HandUp")]
    annot_HD = annotDf.loc[(annotDf["MainCategory"]=="HandDown")]

    # for act in activities:
    #     annot_HU = annotDf.loc[(annotDf["MainCategory"]=="HandUp")&(annotDf["Annotation"]==act)]
    #     annot_HD = annotDf.loc[(annotDf["MainCategory"]=="HandDown")&(annotDf["Annotation"]==act)]
    annot_HD.to_csv("../"+protocol+"/subject/"+subjfolder+"/tmp_HD.csv")
    annot_HU.to_csv("../"+protocol+"/subject/"+subjfolder+"/tmp_HU.csv")

    if(checkHandUpDown(annot_HU, annot_HD)):
        print("non-feeding gesture error")
        exit()
    if len(annot_HU) != len(annot_HD):
        print("hand up and hand down not pairs")
        print(len(annot_HU))
        print(len(annot_HD))
        exit()


    HU_St_list = list(annot_HU.StartTime.tolist())
    HD_Et_list = list(annot_HD.EndTime.tolist())

    UD_dur = []

    for n in range(len(HU_St_list)):
        UD_dur.append([HU_St_list[n],HD_Et_list[n]])

    r_df_test_label = markClassPeriod( r_df_test,'nonfeedingClass' , UD_dur )


    for i,activity in enumerate(activities):

        annot_act = annotDf.loc[(annotDf["Annotation"]==activity)]#&((annotDf["MainCategory"]=="Drinking")|(annotDf["MainCategory"]=="Eating")|(annotDf["MainCategory"]=="Other"))
        act_St_list = list(annot_act.StartTime.tolist())
        act_Et_list = list(annot_act.EndTime.tolist())
        act_dur = []

        for n in  range(len(act_St_list)):
            act_dur.append([act_St_list[n],act_Et_list[n]])

        r_df_test_label = mark( r_df_test_label , 'activity', i+1, act_dur )

        print(act_dur)

    return r_df_test_label


def genFeedingGesture_DrinkingLabels(annotDf, r_df_test, activities):

    # 'feedingClass' = 1 means it is feeding gesture including drinking
    # 'feedingClass' = 0 means it is not feeeding

    # 'activity' implies the activity of the whole period

    annot_feeding = annotDf.loc[annotDf["MainCategory"]=="FeedingGesture"]
    annot_drinking = annotDf.loc[annotDf["MainCategory"]=="Drinking"]

    Feeding_St_list = list(annot_feeding.StartTime.tolist())
    Feeding_Et_list = list(annot_feeding.EndTime.tolist())

    drinking_St_list = list(annot_drinking.StartTime.tolist())
    drinking_Et_list = list(annot_drinking.EndTime.tolist())

    feeding_dur = []
    drinking_dur = []

    for n in range(len(Feeding_St_list)):
        feeding_dur.append([Feeding_St_list[n],Feeding_Et_list[n]])

    for n in range(len(drinking_St_list)):
        drinking_dur.append([drinking_St_list[n],drinking_Et_list[n]])

    r_df_test_label = markClassPeriod( r_df_test,'feedingClass' , feeding_dur )
    r_df_test_label = markClassPeriod( r_df_test_label,'drinkingClass' , drinking_dur )

    for i,activity in enumerate(activities):

        annot_act = annotDf.loc[(annotDf["Annotation"]==activity)]#&((annotDf["MainCategory"]=="Drinking")|(annotDf["MainCategory"]=="Eating")|(annotDf["MainCategory"]=="Other"))
        act_St_list = list(annot_act.StartTime.tolist())
        act_Et_list = list(annot_act.EndTime.tolist())
        act_dur = []

        for n in  range(len(act_St_list)):
            act_dur.append([act_St_list[n],act_Et_list[n]])

        r_df_test_label = mark( r_df_test_label , 'activity', i+1, act_dur )
        print(act_dur)

    return r_df_test_label



def mergeFeatsLabels(allfeatDF, r_df_test_label, activities, lfeatfile):
    
    allfeatDF.index = range(allfeatDF.shape[0])
    allfeatDF['feedingClass'] = 0
    allfeatDF['drinkingClass'] = 0
    allfeatDF['activity'] = 0
    allfeatDF['nonfeedingClass'] = 0

    i = 0
    idx = 0
    # while(i+winsize < r_df_test.shape[0]):    
    while(idx < allfeatDF.shape[0]):    
        class_arr = r_df_test_label.feedingClass[i:i+winsize].as_matrix()
        if Counter(class_arr)[1] > int(winsize/2):
            allfeatDF.ix[idx,"feedingClass"] = 1
        i += stride
        idx += 1

    i = 0
    idx = 0

    while(idx < allfeatDF.shape[0]):    

    # while(i+winsize < r_df_test.shape[0]):    
        class_arr = r_df_test_label.drinkingClass[i:i+winsize].as_matrix()
        if Counter(class_arr)[1] > int(winsize/2):
            allfeatDF.ix[idx,"drinkingClass"] = 1
        i += stride
        idx += 1

    i = 0
    idx = 0
    while(idx < allfeatDF.shape[0]):    
    # while(i+winsize < r_df_test.shape[0]):    
        class_arr = r_df_test_label.activity[i:i+winsize].as_matrix()
        for i_act ,activity in enumerate(activities):

            if Counter(class_arr)[i_act+1] > int(winsize/2):
                allfeatDF.ix[idx,"activity"] = i_act+1
        i += stride
        idx += 1

    i = 0
    idx = 0
    while(idx < allfeatDF.shape[0]):    
    # while(i+winsize < r_df_test.shape[0]):    
        class_arr = r_df_test_label.nonfeedingClass[i:i+winsize].as_matrix()
        for i_act ,activity in enumerate(activities):

            if Counter(class_arr)[i_act+1] > int(winsize/2):
                allfeatDF.ix[idx,"nonfeedingClass"] = i_act+1
        i += stride
        idx += 1

    allfeatDF = allfeatDF[['energy_acc_xyz','orientation_acc_xyz', 'energy_orientation', 'energy_acc_xxyyzz', 'energy_ang_xyz',"energy_ang_xyz_regularized", 'feedingClass', 'drinkingClass', 'activity','nonfeedingClass'
    ]];

    if save_flg:
        allfeatDF.to_csv(lfeatfile)

    return allfeatDF


#  
def mergeFeatsLabels_woAct(allfeatDF, r_df_test_label, lfeatfile):
    
    allfeatDF.index = range(allfeatDF.shape[0])
    allfeatDF['feedingClass'] = 0
    allfeatDF['nonfeedingClass'] = 0

    i = 0
    idx = 0
    # while(i+winsize < r_df_test.shape[0]):    
    while(idx < allfeatDF.shape[0]):    
        class_arr = r_df_test_label.feedingClass[i:i+winsize].as_matrix()
        if Counter(class_arr)[1] > int(winsize/2):
            allfeatDF.ix[idx,"feedingClass"] = 1
        i += stride
        idx += 1

    i = 0
    idx = 0
    while(idx < allfeatDF.shape[0]):    
    # while(i+winsize < r_df_test.shape[0]):    
        class_arr = r_df_test_label.nonfeedingClass[i:i+winsize].as_matrix()
        if Counter(class_arr)[1] > int(winsize/2):
            allfeatDF.ix[idx,"nonfeedingClass"] = 1
        i += stride
        idx += 1

    allfeatDF = allfeatDF[['energy_acc_xyz','orientation_acc_xyz', 'energy_orientation', 'energy_acc_xxyyzz', 'energy_ang_xyz',"energy_ang_xyz_regularized", 'feedingClass', 'nonfeedingClass'
    ]];

    if save_flg:
        allfeatDF.to_csv(lfeatfile)

    return allfeatDF


def genVideoSyncFile(featsfile, birthtime, r_df_test, save_flg, syncfile):
    video_sensor_bias_ms = 0
    featDF = pd.read_csv(featsfile)
    featDF.index = range(featDF.shape[0])

    r_df_test['Time'] = r_df_test.index

    birthtime_s = birthtime[:19]

    import time
    
    base_unixtime = time.mktime(datetime.datetime.strptime(birthtime_s,"%Y-%m-%d %H:%M:%S").timetuple())
    base_unixtime = base_unixtime*1000 + video_sensor_bias_ms

    r_df_test["synctime"] = (r_df_test["synctime"] - base_unixtime)/1000
    extr_idx = list(range(0,len(r_df_test)-winsize,stride))
    r_df_tDsample = r_df_test.iloc[extr_idx]

    r_df_tDsample.index = range(len(r_df_tDsample))

    raw_energy = pd.concat([featDF, r_df_tDsample], axis=1)

    if save_flg:
        raw_energy = raw_energy[['Time','unixtime','synctime','energy_acc_xyz','orientation_acc_xyz','energy_orientation',"energy_acc_xxyyzz",'Angular_Velocity_x','Angular_Velocity_y','Angular_Velocity_z','Linear_Accel_x','Linear_Accel_y','Linear_Accel_z','Class']]
        raw_energy.to_csv(syncfile)

