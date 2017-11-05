import numpy as np

def gen_integ(y):
    y_int = np.zeros(y.size)
    y_int[0] = y[0]
    for i in range(1, y.size):
        y_int[i] = y_int[i-1] + y[i]
        
    return y_int

    
def gen_feat_rec(y, stride = 4, rec = 1):
    # level_1 = 40
    # level_2 = 60
    # level_3 = 80
    # level_4 = 100
    # level_5 = 120
    # l = y.size
    # #make size fixed
    # if l<=level_1:
    #      y=np.interp(np.arange(0, l, l/level_1), np.arange(0, l), y)        
    # elif l>level_1 and l <=level_2:
    #      y=np.interp(np.arange(0, l, l/level_2), np.arange(0, l), y)
    # elif l>level_2 and l <=level_3:
    #      y=np.interp(np.arange(0, l, l/level_3), np.arange(0, l), y)
    # elif l>level_3 and l <=level_4:
    #      y=np.interp(np.arange(0, l, l/level_4), np.arange(0, l), y)
    # elif l>level_4 and l <=level_5:
    #      y=np.interp(np.arange(0, l, l/level_5), np.arange(0, l), y)
    # elif l>level_5 and l <=level_6:
    #      y=np.interp(np.arange(0, l, l/level_6), np.arange(0, l), y)                   
       
    y_integ = gen_integ(y)
    # print(y_integ)
    l = len(y_integ)
    # print('length is',l)
    #feats = np.zeros(435) This is not dynamic programmed, using list instead to make more flexible 
    feats = []

    if rec == 1:
        for i in range(0, l, stride):
            for j in range(i+stride, l, stride):
                feats.append(y_integ[j] - y_integ[i])
    
    elif rec == 2: 
        for i in range(0, l, stride):
            for j in range(i + stride, l, stride):
                for k in range(j + stride, l, stride):
                    tmp1 = y_integ[j] - y_integ[i]
                    tmp2 = y_integ[k] - y_integ[j]
                    feats.append(tmp1 - tmp2*(j - i)/(k - j))
    
    elif rec == 3: 
        for i in range(0, l, stride):
            for j in range(i + stride, l, stride):
                for k in range(j + stride, l, stride):
                    for q in range(k + stride,l,stride):
                        tmp1 = y_integ[j] - y_integ[i]
                        tmp2 = y_integ[k] - y_integ[j]
                        tmp3 = y_integ[q] - y_integ[k]
                        feats.append(tmp1 - tmp2*(j - i)/(k - j)-tmp3*(k - j)/(q - k))
               
        
    feats = np.asarray(feats)          

    return feats



    
def gen_feat_rec_name(y, stride = 4, rec = 1):
    l = len(y)
    # print('length is',l)
    #feats = np.zeros(435) This is not dynamic programmed, using list instead to make more flexible 
    names = []
    if rec == 1:
        for i in range(0, l, stride):
            for j in range(i+stride, l, stride):
                names.append('rec'+str(rec)+'_'+str(i)+'_'+str(j))
    
    elif rec == 2: 
        for i in range(0, l, stride):
            for j in range(i + stride, l, stride):
                for k in range(j + stride, l, stride):
                    names.append('rec'+str(rec)+'_'+str(i)+'_'+str(j)+'_'+str(k))
    
    elif rec == 3: 
        for i in range(0, l, stride):
            for j in range(i + stride, l, stride):
                for k in range(j + stride, l, stride):
                    for q in range(k + stride, l, stride):   
                        names.append('rec'+str(rec)+'_'+str(i)+'_'+str(j)+'_'+str(k)+'_'+str(q))
        
    return names



def resample(signal,unix):
    #every time the input should be one-dimension signal like accx or accy single
    #To make all features resampled, please use a for loop and create an empty dataframe and insert the new value into it every iteration
    unix_new=np.arange(unix[0],(len(unix))*50+unix[0],50)
    off_set=unix_new/unix
    print(off_set)
    new_index=off_set*np.arange(0,len(off_set),1)
    print(new_index)
    new_signal=np.interp(new_index,np.arange(0,len(signal)),signal)
    return new_signal


def resample1(signal,unix):
    #every time the input should be one-dimension signal like accx or accy single
    #To make all features resampled, please use a for loop and create an empty dataframe and insert the new value into it every iteration
    unix_new=np.arange(unix[0],unix[-1],50)
    off_set=unix_new/unix
    new_index=off_set*np.arange(0,len(off_set),1)
    new_signal=np.interp(new_index,np.arange(0,len(signal)),signal)
    return new_signal




#example use of resample
# file = 'accel_label.csv'
# acc_df = pd.read_csv(file)
# ABSOLUTE_TIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f%z"
# acc_df['Time'] = pd.to_datetime(acc_df['Time'])
# y=acc[['accX']].as_matrix().ravel()
# unix=acc[['Unixtime']].as_matrix().ravel()
# y_d_sm = savgol_filter(y, window_length=9, polyorder=2, deriv=1)

# new_signal=resample(y_d_sm,unix)
# features= gen_feat_rec(new_signal,stride = 4,rec=3)


if __name__ == "__main__":

    # resample test case

    # signal = np.array([1,2,3,4,5])
    # unix = [1500000,1500050,1500060,1500100,1500150]
    # print(resample1(signal, unix))
    # output should be [1,2,4,5]

    y = np.ones([150])
    print(y.shape[0])
    for i in range(1,y.shape[0]):
        y[i] = y[i-1]+1
    print(y)

    print(gen_feat_rec(y, stride = 4, rec = 2))




