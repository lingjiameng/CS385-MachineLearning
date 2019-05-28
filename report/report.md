# report

## data preprocessing

### 原始数据和样本边框

				HTML


​					
​				
​				
​						
​				
			lagrange绿色框为正样本，黑色框为负样本，蓝色为原始数据椭圆​

 ### HOG特征图

<div><center><img src="report/2002-07-19-big-img_381-1-neg.jpg" style="zoom:150%;"><img src="report/2002-07-19-big-img_381-1-hog.jpg" style="zoom:150%;"><img src="report/2002-08-14-big-img_1065-7-neg.jpg" style="zoom:150%;"><img src="report/2002-08-14-big-img_1065-7-hog.jpg" style="zoom:150%;"></br><p>随机从处理结果中抽取两个样本,可以看到HOG特征图和原始图像匹配良好</p></center></div>

## Model learning

### logistic model

#### Loss

learning model is to minimize loss function
$$
\begin{aligned}
\sum_{i=1}^n L(&y_i,X_i^T\beta)=\sum_{i=1}^n log[1+exp(-y_iX_i\beta)]\\&where 
\left\{ \begin{aligned}
y_i&:1\times1 \text{ , i-th label with value\{-1,+1\}} \\
X_i&:p\times1 \text{ , i-th data with p features}\\
\beta&:p\times1 \text{ , parameter with p numbers}\\
i&:index \text{, from 1 to n, there n samples}\\
\end{aligned}
\right.
\end{aligned}
$$
actually, $p=900$

#### Gradient

the gradient of loss function is as below:
$$
\begin{align}
\sum_{i=1}^n \frac{\part L(y_i,X_i^T\beta)}{\part \beta}
&=\sum_{i=1}^n \frac{exp(-y_iX_i\beta)}{1+exp(-y_iX_i\beta)}(-y_iX_i)\\
&=\sum_{i=1}^n \frac{1}{1+exp(y_iX_i\beta)}(-y_iX_i)
\end{align}
$$

#### Matrices Forms for  Loss and Gradient

let 
$$
\begin{align}
&X=[X_1^T;X_2^T;...,X_n^T]&n\times p\\
&Y=[y_1,y_2,...,y_n]&n\times 1\\
&\beta &p\times 1 \\
&L = \sum_i^nL(y_i,X_i^T\beta)&1 \times 1 \\
\end{align}
$$
and define a new operation for coding
$$
sum(M) = \sum_i^n m_i\text{ , where }M = [m_1,m_2,...,m_n]
$$
so there is a matrix form for upper equation, **all basic operations are element-wise  except operations only for matrices** :
$$
L=sum(log[1+exp(-Y\circ(X\beta))]) \\
\frac{\part L}{\part \beta}=(\frac{1}{1+exp(Y\circ(X\beta))})^T(Y\circ X)
$$
the **code** are as below

```python
# X shape: n*p
# Y shape: n*1 with{-1,+1}
p = np.multiply(Y, np.matmul(X, self.w)[:, np.newaxis])

# forward loss
loss = np.sum(np.log(1+np.exp(-1*p)))

#gradient
gradient = np.matmul(np.divide(1, 1 + np.exp(p)).T, -np.multiply(Y, X))
```

#### Optimizerlagrange

##### SGD

batch then learn

```python
## SGD
self.w = self.w - self.lr*gradient
```

##### SGLD

```python
## Langevin  SGLD
epsilon = np.random.normal(loc=0.0, 
                           scale=np.sqrt(self.lr), 
                           size=self.n_feature)

self.w = self.w - 0.5*self.lr*gradient + np.sqrt(self.lr)*epsilon
```

#### Result

In the **first epoch**, we can see the loss and accuracy increased a lot.

![logisticlossacc](report/logisticlossacc.png)

the whole process of training reslut is

![epochslossacc](report/epochslossacc.png)

After 50 epoch training, I finally reach the loss $0.06950$ and accuracy $0.97500$ in training set (20680 faces in total),

Using this model, I get $119$ wrong prediction with  correct rate $0.97713$ in test set (5203 faces in total).

the model parameters are shown in below:

![logisticmodelpara](report/logisticmodelpara.png)

we can see that there are more negtive weights than positive weights, since the fact that it need more negtive info to central the bounding box for face rather than loacted in other place

### Fisher Model (LDA)

#### target function

$$
\underset{\beta}{max}\beta^TS_B\beta\text{ ,  s.t.  }\beta^TS_W\beta=1 
$$
Using lagrange multipliers for this maximization problem:
$$
L =\beta^TS_B\beta - \lambda(\beta^TS_W\beta-1)\\
\text{with condition}
\left\{
\begin{aligned}
\frac{\part L}{\part \beta} &=2 S_B\beta-2\lambda S_W\beta = 0\\
\frac{\part L}{\part \lambda}&=0
\end{aligned}
\right.
$$
from condition (1), we can get two information:
$$
S_W^{-1} S_B \beta = \lambda \beta \\
$$
and 
$$
S_B \beta = \lambda S_W \beta
$$
#### 最优解以及特征向量和特征值的分析和讨论

>1)要求解的特征向量只有一个，即与 $S_W^{-1}(\mu^+ - \mu^-)$ 共线的向量。
>
>2)$||\lambda||$代表了目标函数，$||\lambda||$越大，分类效果越好；$||\lambda||$越小，代表数据混合越均匀。

From eqution (10) ,we can get $\beta$ is a eigenvector of matrix $S_W^{-1} S_B \beta$ . Furthermore, we can find that $\beta \propto S_W^{-1}(\mu^+ - \mu^-)$ by:
$$
\begin{aligned}
S_B\beta &= (\mu^+ - \mu^-)(\mu^+ - \mu^-)^T \beta \\
&= (\mu^+ - \mu^-)((\mu^+ - \mu^-)^T \beta)\\
&= (\mu^+ - \mu^-)\alpha\\
S_W^{-1}S_B \beta &= \alpha S_W^{-1}(\mu^+ - \mu^-)\\
&= \lambda \beta \text{ ,   which means }\beta \text{ must be collinear with }S_W^{-1}(\mu^+ - \mu^-)
\end{aligned}
$$

从上面我们可以得出我们要求解的特征向量只有一个，即与 $S_W^{-1}(\mu^+ - \mu^-)$ 共线的向量.

From eqution (11),等式两侧同乘以$\beta^T$,可以得到 $\lambda = \frac{\beta^TS_B\beta}{\beta^TS_W \beta}$ ,所以**$||\lambda||$代表了目标函数**，**$||\lambda||$越大，分类效果越好**。我们同样可以用这个标准来选择特征值和特征向量.

Then I use numpy to calcultate  eigens , results  are as belows:

![ldaeigvalues](report/ldaeigvalues-1558707413008.png)

我们可以看到只有一个特征值数量级为 $10^{-3}$(实数), 其他特征值的数量级全部小于$10^{-15}$(大多为复数). 下图是所有特征值在复平面上的分布

![ldaeigvaluescomplex](report/ldaeigvaluescomplex-1558707526575.png)

根据之前的结论$||\lambda||$代表了分类效果，我们可以得出**只有一个特征值(数量级为$10^{-3}$)相对应的特征向量有分类效果**。其他的特征向量的效果恰恰相反，它们**最小化了目标函数**，使特征分布的类间方差相对于类内方差几乎为0，这意味着**其他的特征向量使两类数据均匀的混合到了一起**，数据分布没有了差异性。

使用复数特征向量变换有些复杂，我选取了全部的实数进行数据降维处理，数据确实均匀的混到了一起，抽选其中的6个展示如下。

![ldamanyres](report/ldamanyres-1558709550100.png)

而特征值(数量级为$10^{-3}$)对应的分类效果与归一化后的$S_W^{-1}(\mu^+ - \mu^-)$分类效果对比如下

![lda2res](report/lda2res.png)

可以看出**分类效果是完全相同的**

同时如下图所示，可以看出这**两个向量也是完全对称的**。（下面第二图是第一图前面一部分的放大结果），这也印证了我们前面从两种不同的形式推出的结论。

![lda2comparisonlocal](report/lda2comparison.png)

![lda2comparisonlocal](report/lda2comparisonlocal.png)

#### 模型分类效果

训练数据集的训练效果如下

训练集类间方差为 $1.61176\times 10^{-6}$，正样本数据集类内方差为 $1.511693\times 10^{-7}$ ，负样本类间方差为$3.893401\times 10^{-8}$. 

当采用不同的值作为分类标准时,结果如下表:

|   分类标准   | 训练集准确率 | 训练集错误数/总数 | 测试集准确率 | 测试集错误数/总数 |
| :----------: | :----------: | :---------------: | :----------: | :---------------: |
|   均值中值   |   0.97950    |     424/20680     |   0.97828    |     113/5203      |
| 正态分布拟合 |   0.97664    |     483/20680     |   0.97847    |     112/5203      |

可以看出准确率大致都在 $97.8$%