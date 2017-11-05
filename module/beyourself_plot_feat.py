import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from beyourself_cascade import read_haar_feat_random_select_samples


if __name__ == "__main__":

    feat_folder = '/Volumes/SHIBO/BeYourself/BeYourself/PROCESS/P120/wrist/haar_feature/'
    meals = ['m0824_1','m0824_2']#

    REC = 12
    WINSIZE = 2



    # ##--------------------------------------------------------------------------------------
    # # SAVE POSITIVE SAMPLES IN A SINGLE FILE
    # XY = read_haar_feat_raw_separate_files(meal, REC, WINSIZE)
    # XYPos =  XY[np.where(XY[:,-1]==1)[0],:]
    # XYNeg =  XY[np.where(XY[:,-1]==0)[0],:]
    # print('XYPos:',XYPos.shape)
    # print('XYNeg:',XYNeg.shape)

    # savename = 'feat_rec'+str(REC)+'_label_win'+str(WINSIZE)+'_pos.txt'
    # np.savetxt(os.path.join(feat_folder, meal, savename), XYPos, delimiter=",")
    # savename = 'feat_rec'+str(REC)+'_label_win'+str(WINSIZE)+'_neg.txt'
    # np.savetxt(os.path.join(feat_folder, meal, savename), XYNeg, delimiter=",")
    # exit()
    # ##--------------------------------------------------------------------------------------



    XY = read_haar_feat_random_select_samples(meals, REC, WINSIZE, ratio = 5, use_seed = 1)


    # Y = XY[:,-1]

    XYPos =  XY[np.where(XY[:,-1]==1)[0],:]
    XYNeg =  XY[np.where(XY[:,-1]==0)[0],:]

    for i in range(XYPos.shape[0]):
        f = plt.figure(figsize=(15,5))
        plt.plot(XYPos[i,:])
        plt.savefig('/Volumes/SHIBO/BeYourself/BeYourself/PROCESS/P120/wrist/m0824_1/pos_rec'+str(REC)+'_win'+str(WINSIZE)+'_'+str(i)+'.png')

    for i in range(XYNeg.shape[0]):
        f = plt.figure(figsize=(15,5))
        plt.plot(XYNeg[i,:])
        plt.savefig('/Volumes/SHIBO/BeYourself/BeYourself/PROCESS/P120/wrist/m0824_1/neg_rec'+str(REC)+'_win'+str(WINSIZE)+'_'+str(i)+'.png')
