import numpy as np
import torch.utils.data


class LogisticClassifier(object):
    def __init__(self,n_feature,vis=True ,batch_size = 64, LR = 0.001):
        self.n_feature = n_feature
        self.lr = LR
        self.vis = vis
        self.batch_size = batch_size
        self.w = np.random.normal(loc=0.0, scale=0.01, size=n_feature)

    def learn(self,X,Y):    
        """
        X shape: n*p
        Y shape: n*1 with{-1,+1}
        """
        # nx1 * (n x p x p x 1)
        p = np.multiply(Y, np.matmul(X, self.w)[:, np.newaxis])

        # forward loss
        loss = np.sum(np.log(1+np.exp(-1*p)))
        p_Y = np.divide(1, 1+np.exp(-p))
        acc = len(p_Y[p_Y > 0.5])

        # backward
        # gradient =np.matmul(np.divide(1, 1 + np.exp(p)).T, -np.multiply(Y, X))

        # gradient = 1-
        gradient = np.matmul(np.divide(1, 1 + np.exp(p)).T, -np.multiply(Y, X))
        gradient = np.squeeze(gradient)
        ## SGD
        self.w = self.w - self.lr*gradient

        # ## Langevin  SGLD
        # epsilon = np.random.normal(loc=0.0, scale=np.sqrt(self.lr), size=self.n_feature)
        # self.w = self.w - 0.5*self.lr*gradient + np.sqrt(self.lr)*epsilon 

        ## lr decay
        # self.lr *=0.9999 

        return loss,acc


    def predict(self,X):
        _Y = 1/(1+np.exp(-np.matmul(X, self.w)))
        
        _Y[_Y <= 0.5] = -1
        _Y[_Y > 0.5] = 1

        _Y = _Y[:,np.newaxis]
        return _Y
