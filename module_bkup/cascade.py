import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pickle




def __weval(yLabel,yPred,w_norm):
    """calculate error in cascade classifier

    Parameters
    ----------
    yLabel: ndarray
    yPred: ndarray
    w_norm: ndarray

    Examples
    --------
    >>>__weval(np.asarray([0,1,0,1,1]),np.asarray([1,1,1,0,1]),np.asarray([0.2,0.1,0.3,.2,.2]))
    0.7
    """

    return np.dot(np.absolute(yLabel - yPred), w_norm)



def __w_beta_update(beta, t, errmin, w_norm, samples, yLabel, yPred):
    """update weights and betas

    step by step manual result for this test code:
	<1>. np.absolute(yLabel - yPred) = [0,1,0]
	<2>. np.ones(samples) - np.absolute(yLabel - yPred) = [1,0,1]
	<3>. np.power(beta[t], np.ones(samples) - np.absolute(yLabel - yPred)) = [0.1111,1,0.1111]
	<4>. w = [ 0.02222222,  0.4       ,  0.04444444]

    Parameters
    ----------
    yLabel: ndarray
    yPred: ndarray
    w_norm: ndarray

    Examples
    --------
    >>>__w_beta_update(beta=np.zeros(10), t=1, errmin=0.1, w_norm=np.array([0.2,0.4,0.4]), samples=3, yLabel=np.array([0,1,0]), yPred=np.array([0,0,0]))
    (array([ 0.02222222,  0.4       ,  0.04444444]),
 	array([ 0.        ,  0.11111111,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ]))

    """

    beta[t] = errmin/(1-errmin)
    w = np.multiply(w_norm, np.power(beta[t], np.ones(samples) - np.absolute(yLabel - yPred)))
    return w, beta



def __pred_weakclf(X,y,d=None):
    """
    Parameters
    ----------
    X: ndarray
    y: ndarray

    Return
    ------
    pred: ndarray

    """

    h = DecisionTreeClassifier(max_depth=1)
    X = X.reshape(-1, 1)
    h.fit(X, y, sample_weight=d)
    pred = h.predict(X)

    return pred


def __save_weakclf(X, y, mdlname, feat, beta, configFile, d=None):
    """save the modified weak classifier model in mdlname, 
    save the index of feature of each weak classifier and beta in configFile

    Parameters
    ----------
    X is one single feature
    feat is the index of this input feature
    beta is cascade coefficient

    """
    h = DecisionTreeClassifier(max_depth=1)
    X = X.reshape(-1, 1)
    h.fit(X, y, sample_weight=d)
    
    # save the model to disk
    pickle.dump(h, open(mdlname, 'wb'))
    
    with open(configFile, 'w') as text_file:
        text_file.write(str(feat))   
        text_file.write('\n')
        text_file.write(str(beta))



def __load__weakclf(mdlname, configFile):
    """load the model from disk

    """
    loaded_model = pickle.load(open(mdlname, 'rb'))
    
    with open(configFile, "r") as f:
        array = []
        for line in f:
            array.append(line)
            
    feat = int(array[0])
    beta = float(array[1])
    
    return loaded_model, feat, beta



def thres_from_betas(betas):
    """load the model from disk

    """
    betas_recip = np.reciprocal(betas)
    alphas = np.log(betas_recip)
    clf_thres = np.sum(alphas)*0.5

    return clf_thres



if __name__ == "__main__":
    print(__w_beta_update(beta=np.zeros(10), t=1, errmin=0.1, w_norm=np.array([0.2,0.4,0.4]), samples=3, yLabel=np.array([0,1,0]), yPred=np.array([0,0,0])))

    # import numpy as np
    # from sklearn.preprocessing import normalize

    # x = np.random.rand(10)*10
    # print(x)
    # norm1 = x / np.linalg.norm(x)
    # norm2 = normalize(x[:,np.newaxis], axis=0, norm = 'l1').ravel()
    # norm3 = normalize(x, norm = 'l1')

    # print(norm1)
    # print(norm2)
    # print(norm3)
    # print (np.all(norm2 == norm1))
