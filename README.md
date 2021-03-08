# Deep-Learning-Tutorials
Self summarized notes from DL by Hung-yi Lee

# Regression

## 0 Introduction

### 分段函数

表示方法：用冲激函数$\delta$作为分类条件的控制门限，并与各项相乘
$$
\delta(state=condition)\tag{1.1}
$$

### Regularization -> Fix Overfitting

$$
L=\sum_n({\hat{y}^n-(b+\sum{w_{i}x_{i}^n)}})^2+\lambda\sum({w_{i}})^2\tag{1.2}
$$

- ***Loss Function***中额外引入$\lambda$一项。$\lambda$越大，则斜率$w$越小（越平滑）的函数被选出
- 正则化时不考虑截距$b$

## 1 Basic Concepts

### Variance

$Def.$ 各函数相较均值的方差

- 简单（低次）模型的受数据影响程度较小，Variance较小
- 复杂模型的Variance较大（测试集误差大）-> Overfitting -> ***More data***（有效但不实际）/ ***Regularization***（将曲线平滑化，可能会使Bias劣化）

### Bias

$Def.$ 均值与最优函数的差距

- 简单模型的Bias较大（不能很好地fit数据点）-> Underfitting -> ***Redesign Model***（增加特征，函数复杂化）
- 复杂模型的Bias较小

### Trade-off

==**本质原因：复杂模型包含简单模型（部分参数置零得到简单模型）**==

### Model Selection

错误方法：不能单纯以Testing set的误差结果作为选择依据，因为Testing set本身存在一定的Bias

#### 解决办法

- ***Cross Validation:*** 将Training set分割成Training set和***Validation set***（验证集）
- ==***N-fold Cross Validation:***== 为避免数据划分错误，将数据集进行N种不同的划分，对每个模型的N种误差求均值，再比较

## 2 Gradient Descent

### Tip1: Tuning Learning Rate $\eta$

#### Adaptive Learning Rates

- 自动调整学习率，逐步变小（与时间相关）

#### Adagrad

- Divide the learning rate of each parameter by the **root mean square** of its previous derivatives' sum（与参数相关）
- 等价于$$\frac{|First\ Derivative|}{\sqrt{Sum\ Of\ (First\ Derivative)^2}}$$

### Tip2: Stochastic Gradient Descent

$$
L^n=({\hat{y}^n-(b+\sum{w_{i}x_{i}^n)}})^2\tag{1.3}
$$

- 只算一个样本的loss，用以提升训练速度

### Tip3: Feature Scaling

- 让不同feature的scale相同，即取值范围相同，让参数更新更容易
- 一种方法：首先计算整体的均值$m_i$和标准差$\sigma_i$，再对于每个特征进行如下运算：

$$
x_i^r\leftarrow\frac{x_i^r-m_i}{\sigma_i}\tag{1.4}
$$

### Gradient Descent Theory

- ***Taylor Series***
- Multi-variable Taylor Series
- ==用一阶泰勒展开简化Loss Function的表达式==
  - 取足够小的圆圈（满足近似条件）
  - 让Loss Function最小，即两个向量反向，且待求向量的模长尽可能大（在圆的边界上）

# Classification

## 0 Introduction

应用：Credit Score, Medical Diagnosis, Handwritten Character Recognition

### Bad Method

- Binary Classification: 可以用Regression的思想解决二分类问题

### Ideal Method

- Loss Function: 训练数据中判断错误的次数

$$
L(f)=\sum_{n}{\delta(f(x^n)\not=\hat{y}^n)}\tag{2.1}
$$

- 寻找最优函数：e.g. **<u>Perceptron</u>**（感知机）, SVM

## 1 Two Classes

### 目标：通过训练数据估测概率，通过概率完成分类

### ==*形式：贝叶斯概率模型*==

$$
P(C_1|x)=\frac{P(x|C_1)P(C_1)}{P(x|C_1)P(C_1)+P(x|C_2)P(C_2)}=\frac{1}{1+exp(-z)}=\sigma(z)\tag{2.2}
$$

- 其中$z=ln\frac{P(x|C_2)P(C_2)}{P(x|C_1)P(C_1)}=…$；==$\sigma(z)$是sigmoid函数，值域为$(0,1)$==
- 若该值大于0.5，则为Class1；否则为Class2

### *生成模型（<u>Generative</u>）*

$$
P(x)=P(x|C_1)P(C_1)+P(x|C_2)P(C_2)\tag{2.3}
$$

- ==假设==样本点服从***Gaussion Distribution***（可以选择其他的），则（后验）概率密度$f_{\mu,\Sigma}(x)$，其中$\mu$代表均值阵，$\Sigma$代表协方差阵：

$$
f_{\mu,\Sigma}(x)=\frac{1}{(2\pi)^{D/2}}\frac{1}{|\Sigma|^{1/2}}exp\left\{{-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)}\right\}\tag{2.4}
$$

#### ***Maximum Likelihood***

- ***性能评价指标***，学习$\mu,\Sigma$，使得$L$越大，则模型越优

$$
L(\mu,\Sigma)=\prod_{i=1}^nf_{\mu,\Sigma}(x^i)\tag{2.5}
$$

### 调整模型

- 令$\mu_1=\mu_2$
- 公用$\Sigma$，取$\Sigma_1$与$\Sigma_2$的加权平均 -> 将$z$化简为线性函数$z=(\mu^1-\mu^2)^T\Sigma^{-1}x-\frac{1}{2}(\mu^1)^T\Sigma^{-1}\mu^1+\frac{1}{2}(\mu^2)^T\Sigma^{-1}\mu^2+ln\frac{N_1}{N_2}$，其中$w^T=(\mu^1-\mu^2)^T\Sigma^{-1}$，$b=-\frac{1}{2}(\mu^1)^T\Sigma^{-1}\mu^1+\frac{1}{2}(\mu^2)^T\Sigma^{-1}\mu^2+ln\frac{N_1}{N_2}$ -> 边界为线性（性能提升）

### 概率分布

1. Gaussion

2. Bernoulli

3. **Naive Bayes Classifier**: All the dimensions are **independent**

## 2 Logistic Regression

### *判别模型（<u>Discriminative</u>）*

$$
f_{w,b}(x)=\sigma(\sum_iw_ix_i+b)\tag{2.6}
$$

### 性能评价

$$
w^*,b^*=arg\,\max_{w,b}L(w,b)\iff w^*,b^*=arg\,\min_{w,b}-lnL(w,b)\tag{2.7}
$$

- 等价运算是为了简化运算
- 其中$L(w,b)=f_{w,b}(x^1)f_{w,b}(x^2)(1-f_{w,b}(x^3))…$，基于假设：其中$x_1,\,x_2$属于Class1，$x_3$属于Class2（所以用1减去概率）
- 将类别用数字表示：==$\hat{y}^n:1\,for\,class\,1,0\,for\,class\,2$==，以便于写进原函数：

$$
-lnL(w,b)=\sum_{i=1}^{n}-[\hat{y}^ilnf(x^i)+(1-\hat{y}^i)ln(1-f(x^i))]\tag{2.8}
$$

- $(2.8)$中每一个求和项都是***==Cross Entrophy== between two Bernoulli distribution：***$H(p,q)=-\sum_{x}p(x)ln(q(x))$

### 寻找最优解

- 对$(2.8)$中每一项对$w_i$求偏微分，得到：

$$
-\frac{\partial lnL(w,b)}{\partial w_i}=\sum_n-(\hat{y}^n-f_{w,b}(x^n))x_i^n\tag{2.9}
$$

$$
w_i\leftarrow w_i-\eta\sum_n-(\hat{y}^n-f_{w,b}(x^n))x_i^n\tag{2.10}
$$

- Logistic Regression的参数更新与Linear Regression的形式一致，只不过$\hat{y}$的取值范围不同
  - Logistic：$\hat{y}\in[0,1]$
  - Linear：$\hat{y}$是真实值
- 如果用Square Error代替Cross Entrophy：离目标很远时，微分值也会近似为零，导致更新很慢，找不到最优解

### Limitations

- 无法处理线性不可分的问题
- 优化方法：
  - Feature transformation: ***heuristic and adhoc***，难以找到合适的transformation
  - Cascading logistic regression models
    - Each logistic regression model -> Neuron
    - Cascading -> Neural Network

## 3 Generative V.S. Discriminative

- Discriminative模型通常优于Generative模型

- ***Generative模型是基于一定假设的***

- 但Generative模型也有一些优点

  1. 所需训练数据较少

  2. 对噪声的鲁棒性更高

  3. 拆分成：Priors probabilities&Class-dependent probabilities

## 4 Multi-Class Classification

### Softmax

- $y_i=e^{z_i}/\sum_{j=1}^ne^{z_j},\;$$\;y_i\in[0,1]$  ***(Maximum Entrophy)***
- 强化最大值，使最大值辨识度更高

### Cross Entrophy

- $\hat{y}^n:n\times1\,vector$：若属于第i类，则第i行的元素为1，其余行为0***（One hot编码）***
- $C(y,\hat{y})=-\sum_{i}\hat{y}_ilny_i$

# Deep Learning

## 0 Introduction

### 发展

- Perceptron -> limitations -> Multi-layer perceptron -> BP -> 1 hidden layer -> 
- RBM initialization -> GPU

### 步骤

#### Define a set of functions -> Neural Network（连接方式，网络结构）

- Fully Connected Feedforward Network
  - Input layer(x-dim), Hidden layers(***Feature extractor***), Output layer(***Multi-Class Classifier***)(y-dim)
  - 矩阵计算：$y=f(x)=\sigma(w^L...\sigma(w^2\sigma(w^1x+b^1)+b^2)...+b^L)$，可通过并行运算提升速度

#### Goodness of function -> Loss Function

- ***Cross Entrophy: $C(y,\hat{y})=-\sum_{i}\hat{y}_ilny_i$（i的最大值等于最终要分成的种类个数）***
- ***Loss Function: $L(\theta)=\sum_{i=1}^NC^i(\theta)$（N的值等于数据个数）***

#### Pick the best function

- Grandient Descent -> Best Parameter Set
- Backpropagation: 一种计算微分的有效方法

### 优点

- Thin&Tall优于Fat&Short -> 原因：

  - Deep -> ***Modularization***（模组化 -> 结构复用）-> 用较少的数据得到较好的结果
  - Image
  - Speech
    - Tri-phone
    - First Step: Classification
      - Gaussion Mixture Model(GMM)
        - Input: acoustic feature
        - Output: state
        - Tied-state
        - Many GMMs
      - Deep Neural Network(DNN)
        - DNN input: One acoustic feature
        - DNN output: Probability of each state
        - One same DNN

- Universality Theorem: Any continuous function can be realized by a network with one hidden layer

  ***But deep structure is more effective***: less parameters and less data required

- ==***End-to-end Learning:*** 只给input和output，中间的每一步复杂过程自动学得（黑盒）==

## 1 Backpropagation

- 待更新参数
  $$
  \theta^k=\theta^{k-1}-\eta\nabla L(\theta^{k-1}),\;\theta=\{w_1,w_2,...,b_1,b_1...\}\tag{3.1}
  $$

- 关键问题：有效地计算$\nabla L(\theta)$

- Chain Rule: 链式求导法则 -> $\frac{\partial C}{\partial w}=\frac{\partial z}{\partial w}\frac{\partial C}{\partial z}$

  Forward pass -> $\frac{\partial z}{\partial w_i}=z_i,\,i.e.\,the \,i\,th\,input$

  **<u>Backward pass</u>** -> $\frac{\partial C}{\partial z}=\sigma^\prime(z)[w_3\frac {\partial C}{\partial z^\prime}+w_4\frac{\partial C}{\partial z^{''}}]$

  <img src="/Users/briankong/Desktop/ML/BP pipline.jpeg" alt="BP pipline" style="zoom:30%;" />

## 2 Tips For Deep Learning

### Recipe of Deep Learning

<img src="/Users/briankong/Desktop/ML/DP pipeline.jpeg" alt="DP pipeline" style="zoom:30%;" />

- 不是所有差的表现都是Overfitting：在Training Data中表现就不好，导致Testing Data中表现不好，是因为本身没有训练好，应考虑更换模型
- $Deeper\not=Better$
- ***Vanishing Gradient Problem:*** Input附近Gradient小，参数更新慢，Output附近参数更新快

### 训练结果不好解决办法

#### New Activation Function

- ***Rectified Linear Unit(ReLU)***
  $$
  \begin{equation}
  a=\left\{
  \begin{aligned}
  & 0 & z\leq0\\
  & z & z\gt0\\
  \end{aligned}
  \;,\;where\,z=wx+b
  \right.
  \end{equation}
  \tag{3.2}
  $$

  - 特点：

    1. 计算快速

    2. 可以解决梯度消失问题

    3. 不可微？输入恰好为0的情况较少，暂不考虑

  - 变式：

    Leaky ReLU
    $$
    \begin{equation}
    a=\left\{
    \begin{aligned}
    & 0.01z & z\leq0\\
    & z & z\gt0\\
    \end{aligned}
    \right.
    \end{equation}
    \tag{3.3}
    $$
    Parametric ReLU
    $$
    \begin{equation}
    a=\left\{
    \begin{aligned}
    & \alpha z & z\leq0\\
    & z & z\gt0\\
    \end{aligned}
    \right.
    \end{equation}
    \tag{3.4}
    $$

- ***Maxout***
  $$
  a=max\{z_1,z_2,...,z_n\}\tag{3.5}
  $$

  - ReLU is a special case of Maxout: $z_1=wx+b\cdot1=wx+b,\;z_2=w\cdot0+b\cdot0=0$

  - **Learnable Activation Function**

  - Training: 不同的输入会生成不同的network structure，保证每一个参数都被train到

#### Adaptive Learning Rate

- ***Adagrad***
  $$
  w^{t+1}\leftarrow w^t-\frac{\eta}{\sqrt{\sum_{i=0}^t(g^i)^2}}g^t\tag{3.6}
  $$
  Use first derivative to estimate second derivative

- ***RMSProp***
  $$
  \begin{equation}
  \begin{aligned}
  & w^1\leftarrow w^0-\frac{\eta}{\sigma^0}g^0,\;where\,\sigma^0=g^0\\
  & w^2\leftarrow w^1-\frac{\eta}{\sigma^1}g^1,\;where\,\sigma^1=\sqrt{\alpha (\sigma^0)^2+(1-\alpha)(g^1)^2}\\
  & \vdots\\
  & \vdots\\
  & w^{t+1}\leftarrow w^t-\frac{\eta}{\sigma^t}g^t,\;where\,\sigma^t=\sqrt{\alpha (\sigma^{t-1})^2+(1-\alpha)(g^t)^2}\\
  \end{aligned}
  \end{equation}
  \tag{3.7}
  $$
  Difficuities: Plateau, Saddle point, Local minima

- ***Momentum***

  Movement is based on both gradient and previous movement
  $$
  \begin{equation}
  \begin{aligned}
  & v^t=\lambda v^{t-1}-\eta \nabla L(\theta^{t-1})\\
  & \theta^{t}=\theta^{t-1}+v^t
  \end{aligned}
  \end{equation}
  \tag{3.8}
  $$

- ***Adam = RMSProp + Momentum***

### 测试结果不好解决办法

#### Early Stopping

- 用Validation Set找到Loss Function的形状，如果是下凸函数，则在极小值点及时停止

#### Regularization

- Loss Function
  $$
  L^\prime(\theta)=L(\theta)+\lambda\frac{1}{2}||\theta||_2\tag{3.9}
  $$
  ***L2 norm***（L2范数）：
  $$
  ||\theta||_2=\sqrt{(w_1)^2+(w_2)^2+...}\tag{3.10}
  $$
  ***L1 norm***（L1范数）：
  $$
  ||\theta||_1=|w_1|+|w_2|+...\tag{3.11}
  $$

- Parameter Update

  ***L2***:
  $$
  w^{t+1}\leftarrow (1-\eta\lambda)w^t-\eta \frac{\partial L}{\partial w}\tag{3.12}
  $$

  - 让第一项参数越来越接近0

  - ***<u>Weight Decay</u>***

  ***L1***:
  $$
  w^{t+1}\leftarrow w^t-\eta\frac{\partial L}{\partial w}-\eta\lambda sgn(w^t)\tag{3.13}
  $$

  - <u>***Sparse***</u>

#### ***Dropout***

- Train每次更新参数前，每个神经元都有p%的概率被Dropout

  - 使得Training的performance变差

- Test时不Dropout，所有权重乘(100 - p)%

  - 使得Testing的performance变好

  - ***Ensemble***: 

    训练一批模型，测试时综合输出

    e.g. Random Forest

## 3 PyTorch Introduction

# Convolutional Neural Network(CNN)

### Basic Concepts（以Image为例）

- ==**<u>Filter</u>**矩阵中的数值是需要通过学习得到的==

- 图像的数学表示

  1. 黑白图像：二维矩阵 -> Filter是二维矩阵

  2. 彩色图像：三层（R，G，B）叠加的立方体阵 -> Filter是立方体阵

### *Pipeline*

<img src="/Users/briankong/Desktop/ML/CNN pipline.jpeg" alt="CNN pipline" style="zoom:30%;" />

#### Convolution

Filter与原图像矩阵根据**<u>Stride</u>**（步长）移动，分别计算**<u>Inner Product</u>**（内积） -> **<u>Feature Map</u>**

- 连接数=Filter矩阵元素个数（删除不必要的连接）-> Less parameters
- ***Shared Weights*** -> Even less parameters

#### Max Pooling

将得到的Inner Product阵划分成多个等大单元，每个单元阵化简为一个值（取平均，取最大值……）

- A new smaller image
- Each filter represents a channel

#### Repeat

- Filter~~个数不会随着重复平方增大，~~只是**维数变大**而已
  - e.g. 第一轮有$25$个$3\times3$的filter，则第二轮有$25$个$3\times3\times25$的filter

- Flatten
  - 将Inner Product阵化为一位向量，通过Fully Connected Network

### CNN in Keras

- Modify network structure
- Input format: vector -> 3-D tensor

### What does CNN learn?

#### Degree of the activation 

$$
a^k=\sum_i\sum_ja_{ij}^k\tag{4.1}
$$

- 利用***Gradient Ascent***，通过改变输入图像x，使得所求值最大
  $$
  \begin{equation}
  x^{*}=\left\{
  \begin{aligned}
  & arg\,\max_x a^k\\
  & arg\,\max_x a^j\\
  & arg\,\max_x y^i\\
  & arg\,\max_x(y^i-\sum_{i,j}|x_{ij}|)
  \end{aligned}
  \right.
  \end{equation}
  \tag{4.2}
  $$

#### Deep Dream

- Let CNN exaggerate what it sees

#### Deep Style

- Make its style like famous paintings

- ***Correlation*** between filters

### Other Applications

#### Playing Go

- Black: 1; White: -1; Blank: 0

- AlhpaGo: Reinforced Learning

#### Speech

- 时-频图像处理

#### Text

- Embedding
  - 每一个单词作为一个vector

# Graph Neural Network(GNN)

## 0 Introduction

### Neural Network

#### CNN

#### RNN

#### Transformer

#### ......

### Graph

#### Node

#### Edge

- 节点集合***V***
- 边集合***E***
- 邻接Neighbor

### GNN: Why

- Classification

- Generation

- 问题？？？

  1. 如何利用结构关系？

  2. 图很大，节点很多怎么办？

  3. 部分数据没有label怎么办？

### GNN: How

- ***Learn From Neighbors***

  1. #### Spatial-based Convolution

  2. #### Spectral-based Convolution

## 1 Roadmap

<img src="/Users/briankong/Desktop/ML/GNN rodamap.jpeg" alt="GNN rodamap" style="zoom:30%;" />

## 2 Spatial-based GNN

- CNN中是N*N方阵；GNN中是相邻的N个邻居
- Terminology
  1. Aggregate: 用***neighbor feature*** update下一层的hidden state
  2. Readout: 把所有nodes的feature集合起来代表整个graph

### NN4G(Neural Network For Graph)

$$
h_i^t=\hat w_{t,t-1}(h_j^{t-i}+h_k^{t-1}+...)\tag{5.1}
$$

把所有与兴趣点相邻的点相加

Readout:
$$
y=\sum_iw_iX_i\tag{5.2}
$$
其中：$X_i=MEAN(h^i)$

### DCNN(Diffusion-Convolution Neural Network)

$$
h_3^0=w_3^0MEAN(d(3,\cdot)=1)\tag{5.3}
$$

把与3号点***距离为1***的点相加取平均
$$
h_3^1=w_3^1MEAN(d(3,\cdot)=2)\tag{5,4}
$$
把与3号点***距离为2***的点相加取平均

- Readout:

  将每一层的feature flatten成一个向量，与W相乘得到Readout

### DGC(Diffusion Graph Convolution)

将每一层的feature直接相加，与W相乘得到Readout

### MoNET(Mixture Model Networks)

Weighted Sum
$$
u(x,y)=(\frac{1}{\sqrt{deg(x)}},\frac{1}{\sqrt{deg(y)}})^T\tag{5.5}
$$

### GraphSAGE

Aggregation: mean, max-pooling, LSTM

### ***GAT(Graph Attention Network)***

学习得到邻居的权重

### GIN(Graph Isomorphism Network)

用Sum替代Max，Min，Mean
$$
h_v^{(k)}=MLP^{(k)}((1+\epsilon^{(k)})\cdot h_v^{(k-1)}+\sum_{u\isin N(v)}h_u^{(k-1)})\tag{5.6}
$$

## 3 Spectral-based GNN

#### 时域卷积 <-> 频域相乘

### Spectral Graph Theory

- Graph: $G=(V,E),\,N=|V|$（节点数量）

- Adjacency Matrix(Weight Matrix)
  $$
  A_{i,j}=0\;if\;e_{i,j}\notin E,\;else\;A_{i,j}=w(i,j)\tag{5.7}
  $$

- ***Undirected*** -> A is symmetric to the diagonal

- Degree Matrix
  $$
  \begin{equation}
  D_{i,j}=\left\{
  \begin{aligned}
  & d(i) & if\;i=j\\
  & 0 & if\;i\not= j\\
  \end{aligned}
  \right.
  \end{equation}
  \tag{5.8}
  $$
  $d(i)$代表邻居个数

- Signal on Graph

  $f:V\rightarrow R^N$, $f(i)$

![Graph Theory 1](/Users/briankong/Desktop/ML/Graph Theory 2.jpeg)

- 将$\lambda$视作频率，频率越大，相邻两点间变化量越大

  #### ***Goal:***

  $$
  y=g_\theta(U\Lambda U^T)x=g_\theta(L)x\tag{5.9}
  $$

  - 其中$g_\theta$表示filter，为***待学习参数***

  - Problems:

    - **Learning complexity depends on the size of input graph**

    - Not Localize

      由于$g_\theta$可以是任意函数，所以可能会学到不需要的东西：e.g. $L^N$会使得节点共享信号信息

### ChebNet

#### Efficient & Localize

##### Use polynomial to parametrize $g_\theta$

$$
g_\theta(L)=\sum_{k=0}^K\theta_kL^k\tag{5.10}
$$

- Problem: complexity of $O(
  N^2)$

  - Solution: **Chebyshev polynomial**
    $$
    \begin{split}
    & T_0(\widetilde{\Lambda})=1,\;T_1(\widetilde{\Lambda})=\widetilde{\Lambda},\;T_k(\widetilde{\Lambda})=2\widetilde{\Lambda}T_{k-1}(\widetilde{\Lambda})-T_{k-2}(\widetilde{\Lambda})\\
    & where\;\widetilde{\Lambda}=\frac{2\Lambda}{\lambda_{max}}-1,\;\widetilde{\lambda}\isin[-1,1]
    \end{split}
    \tag{5.11}
    $$

    $$
    g_{\theta^\prime}(\widetilde\Lambda)=\sum_{k=0}^K\theta_k^\prime T_k(\widetilde\Lambda)\tag{5.12}
    $$

    替换后 -> 简化计算

- 结果
  $$
  y=g_{\theta^\prime}(L)x=[x_0\;x_1\;...\;x_K][\theta_0^\prime\;\theta_1^\prime\;...\;\theta_K^\prime]\tag{5.13}
  $$

### ***GCN***

- 原理
  $$
  \begin{split}
  y & =g_{\theta^\prime}(L)x=\sum_{k=0}^K\theta_k^\prime T_k(\widetilde L)x,\;K=1\\
  & =\theta(I+D^{-\frac{1}{2}}AD^{\frac{1}{2}})x
  \end{split}
  \tag{5.14}
  $$

- Renormalization trick
  $$
  H^{(l+1)}=\sigma(\widetilde D^{-\frac{1}{2}}\widetilde A\widetilde D^{\frac{1}{2}}H^{(l)}W^{(l)})\tag{5.15}
  $$

- ***核心***

$$
h_v=f(\frac{1}{|N(v)|}\sum_{u\isin N(v)}Wx_u+b)\tag{5.16}
$$

## 4 Graph Gneration

### VAE

### GAN

### Auto-regressive-based
