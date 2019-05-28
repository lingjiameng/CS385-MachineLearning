import numpy as np

#for CNN
import torch 
import torch.nn as nn


class LogisticClassifier(object):
    def __init__(self,n_feature,vis=True ,batch_size = 64, LR = 0.001,loss="SGD"):
        self.n_feature = n_feature
        self.lr = LR
        self.vis = vis
        self.batch_size = batch_size
        self.w = np.random.normal(loc=0.0, scale=0.01, size=n_feature)
        self.loss = loss

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

        if self.loss == "SGD":
            ## SGD
            self.w = self.w - self.lr*gradient
        elif self.loss == "SGLD":
            # ## Langevin  SGLD
            epsilon = np.random.normal(loc=0.0, scale=np.sqrt(self.lr), size=self.n_feature)
            self.w = self.w - 0.5*self.lr*gradient + np.sqrt(self.lr)*epsilon 
        else :
            raise "no such loss funtion"

        ## lr decay
        # self.lr *=0.9999 

        return loss,acc


    def predict(self,X):
        _Y = 1/(1+np.exp(-np.matmul(X, self.w)))
        
        _Y[_Y <= 0.5] = -1
        _Y[_Y > 0.5] = 1

        _Y = _Y[:,np.newaxis]
        return _Y

class LDA(object):
    def __init__(self,vis=False):
        self.vis = vis
        self.w = None
        self.mu_pos = 0.0
        self.var_pos = 1.0
        self.mu_neg = 0.0
        self.var_neg = 1.0

    def _normpdf(self,mu,cov,x):
        ##只适用于一维变量的分布
        ## x 可以为向量
        scalar = 1.0/np.sqrt(2*np.pi*cov)
        exp = np.exp(-1.0/(2*cov)*(x-mu)**2)
        return scalar*exp

    def learn(self,X,Y):
        """
        X shape: n*p
        Y shape: n*1 with{-1,+1}
        """
        data_pos = X[np.squeeze(Y == 1), :]
        data_neg = X[np.squeeze(Y == -1), :]

        # get mean and covariance
        mu_pos = data_pos.mean(axis=0).reshape(-1, 1)
        mu_neg = data_neg.mean(axis=0).reshape(-1, 1)
        cov_pos = np.cov(data_pos, rowvar=False)
        cov_neg = np.cov(data_neg, rowvar=False)

        # get s_B and s_W
        delta_mu = mu_pos-mu_neg
        SB = np.matmul(delta_mu, delta_mu.T)
        SW = (len(data_pos)*cov_pos + len(data_neg)*cov_neg)
        SW_inv = np.linalg.inv(SW)

        # get self.w
        self.w = np.matmul(SW_inv, delta_mu)

        # get train data projection location
        z = np.matmul(X, self.w)

        #get mean and variance from projection result
        self.var_pos = np.var(z[Y==1])
        self.mu_pos  = np.mean(z[Y==1])
        self.var_neg = np.var(z[Y==-1])
        self.mu_neg  = np.mean(z[Y==-1])

        #对比 以不同方式计算出来的 的方差精度
        # z_cov_pos = np.matmul(self.w.T, np.matmul(cov_pos, self.w))
        # z_mu_pos = np.matmul(mu_pos.T, self.w)
        # z_cov_neg = np.matmul(self.w.T, np.matmul(cov_neg, self.w))
        # z_mu_neg = np.matmul(mu_neg.T, self.w)
        # print("pos:[var {:.5e} cov {:.5e}]".format(self.var_pos,np.squeeze(z_cov_pos)))
        # print("pos:[mu  {:.5e} mu  {:.5e}]".format(self.mu_pos,np.squeeze(z_mu_pos)))
        # print("neg:[var {:.5e} cov {:.5e}]".format(self.var_neg, np.squeeze(z_cov_neg)))
        # print("neg:[mu {:.5e} mu {:.5e}]".format(self.mu_neg, np.squeeze(z_mu_neg)))

        # 根据正态分布计算处的概率密度判断类别
        z_norm_pos = self._normpdf(self.mu_pos, self.var_pos, z)
        z_norm_neg = self._normpdf(self.mu_neg, self.var_neg, z)
        z_norm = np.zeros(z.shape)
        z_norm[z_norm_neg >= z_norm_pos] = -1
        z_norm[z_norm_neg < z_norm_pos] = 1

        # #根据均值中点判断类别
        # pos = (self.mu_pos > self.mu_neg)*2-1
        # mu_mid = (self.mu_neg+self.mu_pos)/2.0
        # z_norm = np.zeros(z.shape)
        # z_norm[z >= mu_mid] = pos
        # z_norm[z < mu_mid]  = -pos

        if self.vis == True:
            print("========================training result========================")
            intra_var = np.squeeze(np.matmul(delta_mu.T, self.w))**2
            inter_var_pos = self.var_pos
            inter_var_neg = self.var_neg
            print("[intra-calss variance: {:.5e}] \
                    \n[inter-class variance for positive: {:5e}]\
                    \n[inter-class variance for negative: {:5e}]"\
                    .format(intra_var,inter_var_pos,inter_var_neg)
            )
            error = np.sum(np.abs(z_norm - Y))/2
            print("[correct rate : {:.5f}][total size: {}][error size: {}]".format(
                1.0-error/len(z), len(z), error))
            print("===============================================================")
        
            

    def predict(self,X):
        # get train data projection location
        z = np.matmul(X, self.w)
        
        #根据正态分布判断类别
        z_norm_pos = self._normpdf(self.mu_pos, self.var_pos, z)
        z_norm_neg = self._normpdf(self.mu_neg, self.var_neg, z)
        z_norm = np.zeros(z.shape)
        z_norm[z_norm_neg >= z_norm_pos] = -1
        z_norm[z_norm_neg < z_norm_pos] = 1

        # #根据均值中点判断类别
        # pos = (self.mu_pos > self.mu_neg)*2-1
        # mu_mid = (self.mu_neg+self.mu_pos)/2.0
        # z_norm = np.zeros(z.shape)
        # z_norm[z >= mu_mid] = pos
        # z_norm[z < mu_mid] = -pos

        return z_norm


class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv = nn.Sequential(   #input shape(1,96,96)
            nn.Conv2d(
                in_channels = 1, #input channel
                out_channels = 16, #output channel
                kernel_size=5, # filter size
                stride=1, # 
                padding=2 # output shape (16,96,96)
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), #池化(16,48,48)
        )
        self.out = nn.Sequential(
            nn.Linear(16*48*48, 2),
        )

    def forward(self,x):
        x = self.conv(x)
        x = x.view(x.size(0),-1)
        output = self.out(x)
        return output , x

