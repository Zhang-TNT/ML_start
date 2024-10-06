# Chapter_1

## Intro

ML指**通过计算结果不断调整算法以达到最有效果**，在数据中产生“模型”的算法，学习任务可划分为两大类：“监督学习”和“无监督学习”，**分类和回归是监督学习的代表**，**聚类是无监督学习的代表**。ML的目的是使产生的模型很好地适用于“新样本”

## Data

这里引用`sklearn`包中`datasets`模块中的`Iris`数据集，根据花瓣的长度和宽度以及花萼的长度和宽度进行分类，调用代码如下

```python
from sklearn.datasets import load_iris

# 导出Iris数据集，load_iris()返回的iris对象是一个Bunch对象，类似字典
iris_dataset = load_iris()

if __name__ == '__main__':
    # 输出数据集键
    print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))
```

输出结果如下

```python
dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])
```

各键所对应的值如下

+ `data`：原始数据，数据结构为`<class 'numpy.ndarray'>`，数据大小为（150，4），每一行对应一朵花，列代表每朵花的四个测量数据，这里给出前5夺花的数据

    ```python
    [[5.1 3.5 1.4 0.2]
     [4.9 3.  1.4 0.2]
     [4.7 3.2 1.3 0.2]
     [4.6 3.1 1.5 0.2]
     [5.  3.6 1.4 0.2]]
    ```

+ `target`：是测量过的每朵花的品种，0代表setosa，1代表versicolor， 2 代表virginica，也即标签

+ `target_names`：花的品种

    ```pyton
    ['setosa' 'versicolor' 'virginica']
    ```

+ `DESCR`：数据集的简要说明

+ `feature_names`：特征说明

    ```python
    ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    ```

+ `filename`：文件名称

+ `data_module`：数据模块

这里给出数据集的简要说明

```python
.. _iris_dataset:

Iris plants dataset
--------------------

**Data Set Characteristics:**

:Number of Instances: 150 (50 in each of three classes)
:Number of Attributes: 4 numeric, predictive attributes and the class
:Attribute Information:
    - sepal length in cm
    - sepal width in cm
    - petal length in cm
    - petal width in cm
    - class:
            - Iris-Setosa
            - Iris-Versicolour
            - Iris-Virginica

:Summary Statistics:

============== ==== ==== ======= ===== ====================
                Min  Max   Mean    SD   Class Correlation
============== ==== ==== ======= ===== ====================
sepal length:   4.3  7.9   5.84   0.83    0.7826
sepal width:    2.0  4.4   3.05   0.43   -0.4194
petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)
petal width:    0.1  2.5   1.20   0.76    0.9565  (high!)
============== ==== ==== ======= ===== ====================

:Missing Attribute Values: None
:Class Distribution: 33.3% for each of 3 classes.
:Creator: R.A. Fisher
:Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)
:Date: July, 1988

The famous Iris database, first used by Sir R.A. Fisher. The dataset is taken
from Fisher's paper. Note that it's the same as in R, but not as in the UCI
Machine Learning Repository, which has two wrong data points.

This is perhaps the best known database to be found in the
pattern recognition literature.  Fisher's paper is a classic in the field and
is referenced frequently to this day.  (See Duda & Hart, for example.)  The
data set contains 3 classes of 50 instances each, where each class refers to a
type of iris plant.  One class is linearly separable from the other 2; the
latter are NOT linearly separable from each other.

|details-start|
**References**
|details-split|

- Fisher, R.A. "The use of multiple measurements in taxonomic problems"
  Annual Eugenics, 7, Part II, 179-188 (1936); also in "Contributions to
  Mathematical Statistics" (John Wiley, NY, 1950).
- Duda, R.O., & Hart, P.E. (1973) Pattern Classification and Scene Analysis.
  (Q327.D83) John Wiley & Sons.  ISBN 0-471-22361-1.  See page 218.
- Dasarathy, B.V. (1980) "Nosing Around the Neighborhood: A New System
  Structure and Classification Rule for Recognition in Partially Exposed
  Environments".  IEEE Transactions on Pattern Analysis and Machine
  Intelligence, Vol. PAMI-2, No. 1, 67-71.
- Gates, G.W. (1972) "The Reduced Nearest Neighbor Rule".  IEEE Transactions
  on Information Theory, May 1972, 431-433.
- See also: 1988 MLC Proceedings, 54-64.  Cheeseman et al"s AUTOCLASS II
  conceptual clustering system finds 3 classes in the data.
- Many, many more ...

|details-end|

```

## Evaluation

使用`Iris`数据构建一个机器学习模型，用于预测新的样本数据，先将该数据集划分为训练数据和测试数据，这里使用函数`train_test_split`将$75\%$的数据作为训练集，余下作为测试集，这一比例是很好的经验法则。

```python
# 划分训练集和测试集，其中X表示数据，y表示标签
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

```

函数`train_test_split`使用伪随机数生成器将数据集打乱，数据是按标签进行排序，参数`random_state`指定随机数生成器的种子。下面对数据进行可视化观察

+ Investigation

    建模之前对数据进行审查，查看是否完整，可视化数据集，一种是散点图，只能一次进行两个特征的绘制；一种是散点图矩阵，可两两查看特征

    + 核概率密度估计

        反映数据集中特征变量分布情况

        ![avator](F:\stuff\gitts\ML\pics\1_data_KDE.png)

    + 面积图

        数据集四特征变量在一个图上分布

        ![avator](F:\stuff\gitts\ML\pics\1_data_AREA.png)

    + 直方图

        数据集四特征变量在四个图上分布

        ![avator](F:\stuff\gitts\ML\pics\1_data_HISAT.png)

    + 两两特征分布

        数据集两两特征变量分布

        ![avator](F:\stuff\gitts\ML\pics\1_data_PAIRPLOT.png)

+ Modeling

    + K-nearest neighbor(K-NN)
    
        K近邻（KNN）属于监督学习算法，给定测试样本，根据**已确定距离度量函数**计算测试样本与所有训练样本的距离，随后找出离测试样本最近的K各训练样本，测试样本的类别就根据这K个训练样本信息决策，即根据计算两个样本各特征值之间的距离进行分类，仅仅需要参考该样本最近的K各样本属于的类别，相当于投票法。影响K近邻分类有三个因素：K的取值、距离度量、分类决策规则。KNN算法在`sklearn`包中的`neighbors`模块的`KNeighborsClassifier`类中实现的，具体如下
    
        ```python
        class sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, weights=’uniform’, 
        											algorithm=’auto’, leaf_size=30, 
        											p=2, metric=’minkowski’, 
        											metric_params=None, 
        											n_jobs=None, **kwargs)
        # 参数n_neighbors	即为k值，默认为5
        # 参数weights		即为权重
        # 参数algorithm	即为搜索算法
        # 参数leaf_size	即为构造树才使用的参数
        # 参数p		    即为距离度量公式，默认为2
        # 参数metric		即为距离度量
        # 参数metric_params	即为距离公式的其他关键参数
        # 参数n_jobs		即为并行处理设置
        ```
    
        想要基于训练集来拟合模型，调用knn对象的fit方法，输入训练集和训练标签，返回knn对象本身并作原处修改（字符串表示），也即返回构建模型时用到的参数
    
        ![avator](F:\stuff\gitts\ML\pics\1_data_KNN.png)
    
    + 做出预测
    
        这里创建一个新数据：`np.array([[5, 2.9, 1, 0.2]])`，使用方法`predict()`，即可返回预测结果
    
        ![avator](F:\stuff\gitts\ML\pics\1_data_PREDICTION.png)
    
    + 评估模型
    
        评估模型需要用到之前创建的测试集，通过预测结果与标签进行对比，计算**预测精度**评估模型，也可以使用方法`score`进行评估
    
        ![avator](F:\stuff\gitts\ML\pics\1_data_PREDICTION_score.png)

# Chapter_2

## 2.1 分类与回归

监督学习主要有**分类**和**回归**，分类问题目标是预测**类别标签**，回归任务目标是预测**连续值**。

## 2.2 评估方法

训练模型能对新数据进行准确预测，说明该模型**泛化精度**高。训练误差称为经验误差，对训练样本学得太好了出现过拟合，期望找到最简单的模型实现对新数据的精准预测。

![avator](F:\stuff\gitts\ML\pics\2_2_1.png)

使用“测试集”对模型进行验证

+ 留出法

    直接将数据集$D$划分为两个互斥的集合，训练集$S$和测试集$T$

+ 交叉验证法

    先将数据集$D$划分为$k$个大小相似的互斥子集$D_i$，每次用$k-1$个子集的并集作为训练集，余下的那个子集作为测试集，这样就可获得$k$组训练/测试集，即可进行$k$次训练和测试，最终返回的是这$k$个测试结果的均值。通常称交叉验证法为“$k$折交叉验证”，同时数据集的划分方法有多种

+ 自助法

    留出法和交叉验证法的训练集都比$D$小，必然会引入估计偏差。自助法在数据集较小、数据集难以划分时很有效，但改变了初始数据集的分布，会引入估计偏差

+ 调参与最终模型

    对算法参数进行设定

## 2.3 性能度量

衡量模型泛化能力的评价标准，常见的有“均方误差”

+ 错误率与精度

    错误率是**分类错误的样本数占样本总数的比例**，精度是**分类正确的样本数占样本总数的比例**

+ 查准率与查全率

    ![avator](F:\stuff\gitts\ML\pics\2_3_1.png)

    查全率与查准率是一对矛盾的度量，更常用的是$F1$度量：$F1=\frac{2\times P\times R}{P+R}=\frac{2\times TP}{样例总数+TP-TN}$

+ ROC与AUC

    $ROC$曲线是研究学习器泛化性能的工具，根据学习器的预测结果对样例进行排序，按此顺序逐个把样本作为正例进行预测，每次计算出两个重要量的值作为横纵轴作图，即得到$ROC$曲线。$AUC$可通过对$ROC$曲线下各部分面积求和而得

+ 代价敏感错误率与代价曲线

## 2.4 比较检验

+ 假设检验
+ 交叉验证$t$检验
+ McNemar检验

## 2.5 监督学习算法

### 2.5.1 一些样本数据集

这里使用一些小规模数据集来说明不同的算法

+ `forge`数据集

    小型模拟分类问题数据集，有26个数据点，每个数据有两个特征

    ![avator](F:\stuff\gitts\ML\pics\2_5_1.png)

+ `wave`数据集

    只有一个输入特征和一个连续的目标变量，形状为$(40, 1)$，这里画出其在$xy$平面上的分布情况

    ![avator](F:\stuff\gitts\ML\pics\2_5_2.png)

+ `breast_cancer`数据集

    使用函数`load_breast_cancer()`返回一个`breast_cancer`数据的`Bunch`对象，该数据集形状为$(569, 30)$，共569个数据点，每个数据点有30个特征，打印其键

    ![avator](F:\stuff\gitts\ML\pics\2_5_3.png)

    特征如下

    ![avator](F:\stuff\gitts\ML\pics\2_5_11.png)

    在569个数据点中，212个被标记为恶性，357个被标记为良性，由于特征非常多，这里选取相近的特征进行可视化

    + 肿瘤半径

        选取`['mean radius', 'radius error', 'worst radius', 'target']`作为构图元素

        ![avator](F:\stuff\gitts\ML\pics\2_5_12.png)

    + 肿瘤质地

        选取`['mean texture', 'texture error', 'worst texture', 'target']`作为构图元素

        ![avator](F:\stuff\gitts\ML\pics\2_5_13.png)

    + 肿瘤周长

        ![avator](F:\stuff\gitts\ML\pics\2_5_14.png)

    + 肿瘤面积

        ![avator](F:\stuff\gitts\ML\pics\2_5_15.png)

    + 肿瘤平滑度

        ![avator](F:\stuff\gitts\ML\pics\2_5_16.png)

    + 肿瘤致密性

        ![avator](F:\stuff\gitts\ML\pics\2_5_17.png)

    + 肿瘤凹度

        ![avator](F:\stuff\gitts\ML\pics\2_5_18.png)

    + 肿瘤凹点

        ![avator](F:\stuff\gitts\ML\pics\2_5_19.png)

    + 肿瘤对称度

        ![avator](F:\stuff\gitts\ML\pics\2_5_20.png)

    + 肿瘤分形维数

        ![avator](F:\stuff\gitts\ML\pics\2_5_21.png)

    由以上特征分布图可知，肿瘤与特征确有某种相关关系

+ `california_housing`数据集

    加州住房数据集，形状为$(20640, 8)$，共20640个数据点，每个数据点有8个特征，打印其键

    ![avator](F:\stuff\gitts\ML\pics\2_5_4.png)

### 2.5.2 K近邻

K近邻（KNN）属于监督学习算法，给定测试样本，根据**已确定距离度量函数**计算测试样本与所有训练样本的距离，随后找出离测试样本最近的K各训练样本，测试样本的类别就根据这K个训练样本信息决策，即根据计算两个样本各特征值之间的距离进行分类，仅仅需要参考该样本最近的K各样本属于的类别，相当于投票法。影响K近邻分类有三个因素：K的取值、距离度量、分类决策规则。

对于二维数据集，可以在$xy$平面上画出所有可能的测试点的预测结果，对类别平面进行染色以查看**决策边界**，这里分别设置$k=1$、$k=3$和$k=9$三种情况

![avator](F:\stuff\gitts\ML\pics\2_5_5.png)

由结果图可以看出，随着邻居个数的增多，决策边界也越来越平滑。这里在乳腺癌数据集上进行测试，以查看训练集精度和测试集精度

![avator](F:\stuff\gitts\ML\pics\2_5_6.png)

$k$近邻算法用于回归，在`sklearn.neighbors`的`KNeighborsRegressor`类中实现，这里以`wave`数据集为案例

![avator](F:\stuff\gitts\ML\pics\2_5_7.png)

由图可以看出，随着邻居个数的增多，预测结果变得更加平滑，但对训练数据集的拟合也不好，$KNN$虽容易理解但预测速度慢且不能处理多特征数据集。

# Chapter_3

## 3.1 线性模型

线性模型利用输入特征的**线性函数**进行预测

### 3.1.1 用于回归的线性模型

对给定数据集，学得一个线性模型。例如：$f(x_i)=\sum_i \omega_ix_i+b$，其中$x_i$表示单个数据点的特征，参数$\omega_i$和$b$是学习模型的参数，这里以`wave`数据集为案例使用代码`mglearn.plots.plot_linear_regression_wave()`即可完成一阶线性模型拟合，相应的参数为$w[0]: 0.393906\quad b: -0.031804$

![avator](F:\stuff\gitts\ML\pics\2_5_8.png)

用于回归的线性模型对于单一特征的预测结果是一条直线，两个特征则是一个平面，更多特征时则是一个超平面。

### 3.1.2 线性回归

线性回归，又称**普通最小二乘法**（OLS），即寻找参数使得对训练集的预测值与真实值之间的**均方误差**最小，使用`sklearn.linear_model`中的`LinearRegression`类可生成线性回归模型，这里以`wave`数据集为案例

![avator](F:\stuff\gitts\ML\pics\2_5_9.png)

可以看到结果不是很好，可能存在欠拟合。这里换用更高维数据集，加州住房数据集，该数据集形状$(20640, 8)$

![avator](F:\stuff\gitts\ML\pics\2_5_10.png)

### 3.1.3 岭回归

标准线性回归最常用的替代方法之一就是**岭回归**，其预测公式与普通最小二乘法相同，但对系数$\omega$的选择不仅要在训练数据上得到好的预测结果，还要**拟合附加约束**，模型拟合过程中希望系数$\omega$尽可能小，即每个特征对输出的影响尽可能小同时仍能给出很好的预测结果，这种约束是**正则化**的一个例子，正则化是指对模型做显式约束以避免出现过拟合，岭回归使用的被称为**L2正则化**，在`linear_model.Ridge`中实现

![avator](F:\stuff\gitts\ML\pics\3_1_1.png)

由图可以看出，8个特征变量仍然是太少，这里换为乳腺癌数据，形状为$(569, 30)$，结果如下

![avator](F:\stuff\gitts\ML\pics\3_1_2.png)

可以看出更大的`alpha`参数表示约束更强的模型，现通过固定`alpha`参数并改变训练量，将模型性能作为数据集大小的函数进行绘图，图像称为**学习曲线**

![avator](F:\stuff\gitts\ML\pics\3_1_3.png)

正则化的测试分数更高，特别是较小数据集，如果足够多的训练数据，正则化变得不重要

### 3.1.4 lasso

Lasso也是一种正则化的线性回归，称作**L1正则化**，其结果是某些系数刚好为0，即只使用少数几个特征，这里以乳腺癌数据集为案例进行`lasso`线性回归

![avator](F:\stuff\gitts\ML\pics\3_1_4.png)

随着参数`alpha`的减小，得到正则化很弱的模型。在实践中一般首选岭回归，如果特征很多且其中几个是重要的即可选择`Lasso`回归

### 3.1.5 用于分类的线性模型

二分类模型可表示为$f(x_i)=\sum_i \omega_ix_i>0$，根据数据是否满足不等式进行分类，最常见的两种线性分类算法是**Logistic回归**和**线性支持向量机**，这里以`forge`数据集为案例

![avator](F:\stuff\gitts\ML\pics\3_1_5.png)

对于这两个方法，决定正则化强度的权衡参数叫作`C`，`C`越大对应的正则化越弱，能尽可能适应大多数的数据点，这里换高维数据集乳腺癌数据集为案例

![avator](F:\stuff\gitts\ML\pics\3_1_6.png)

用于分类的线性模型和用于回归的线性模型有许多相似之处，主要差别在于惩罚参数

### 3.1.6 用于多分类的线性模型

将二分类算法推广到多分类算法的一种常见方法是**一对其余**方法，对每个类别都学习一个二分类模型，在测试点上运行所有二类分类器来进行预测，每个类别都有一个系数$\omega$向量和一个截距$b$，这里使用`make_blobs`一个三分类数据集，每个类别的数据都是从一个高斯分布中采样得到

![avator](F:\stuff\gitts\ML\pics\3_1_7.png)

在这个数据集上训练一个`LinearSVC`分类器，并将分类器可视化

![avator](F:\stuff\gitts\ML\pics\3_1_8.png)

这里给出二维空间中所有区域的预测结果

![avator](F:\stuff\gitts\ML\pics\3_1_9.png)

### 3.1.7 优点、缺点和参数

线性模型的主要参数是**正则化**参数，通常在对数尺度上对这些参数进行搜索，另外就是**L1正则化**和**L2正则化**的确定。线性模型的训练速度非常快，预测速度也很快

## 3.2 朴素贝叶斯分类器

朴素贝叶斯分类器相比线性模型训练速度更快，但代价即是泛化能力稍差，其通过单独查看每个特征来学习参数，并从每个特征中收集简单的类别统计数据，`sklearn`中有三种朴素贝叶斯分类器，分别是`GaussianNB`、`BernoulliNB`和`MultinomialNB`，其中`GussianNB`可应用于任意连续数据，`BernoulliNB`假定输入数据为二分类数据，`MultinomialNB`假定输入数据为计数数据，后两种主要用于文本数据分类。

+ `BernoulliNB`分类器

    计算每个类别中每个特征不为0的元素个数

+ `MultinomialNB`分类器

    计算每个类别中每个特征的平均值

+ `GaussianNB`分类器

    会保存每个类别中每个特征的平均值和标准差

将数据点与每个类别的统计数据进行比较，将最匹配的类别作为预测结果，朴素贝叶斯模型是很好的基准模型，常用于非常大的数据集

对于“分类问题”，可使用“对数几率函数”替代单位阶跃函数，同样是利用线性回归模型的预测结果逼近真实标记的对数几率

## 3.4 线性判别分析

线性判别分析（LDA）是一种经典的线性学习方法，给定训练样例集，设法将样例投影到一条直线上。使得同类样例的投影点尽可能接近、异类样例的投影点尽可能远离

## 3.5 多分类学习

多分类的基本思路是“拆解法”，将多分类任务拆解为若干个二分类任务

## 3.6 类别不平衡问题

类别不平衡指分类任务中不同类别的训练样例数目差别很大的情况，不失一般性，需要进行“再缩放”或者“再平衡”

# Chapater_4

## 4.1 决策树

决策树（decision tree）是一类常见的机器学习方法，基于树结构进行决策，广泛用于分类和回归任务的模型，本质上从一层层的if/else问题中进行学习并得出结论。例如对“这是好瓜吗”进行决策，决策过程如下图

![avator](F:\stuff\gitts\ML\pics\4_1_1.png)

一般的，一棵决策树包含一个根结点、若干内部结点和若干叶结点，其目的是产生一颗泛化能力强的决策树，基本流程遵循“分而治之”策略

![avator](F:\stuff\gitts\ML\pics\4_1_2.png)

由图可看出决策树的生成是一个递归过程

## 4.2 构造决策树

决策树学习的关键是如何进行最优划分属性，用于连续数据的测试形式是：特征$i$的值是否大于$a$，决策树的每个结点都包含一个测试，每个测试可视为对沿着一条轴对当前数据进行划分，即分层划分的观点，对数据反复进行递归划分，直到划分后的每个区域只包含单一目标值。对新数据点的预测，先查看这个点位于特征空间划分的所在区域，然后将该区域的多数目标值作为预测结果，从根节点开始对树进行遍历。

## 4.3 控制决策树复杂度

通常来说构造决策树直到所有叶结点都是纯的叶结点会导致模型非常复杂，还会出现过拟合现象，剪枝处理是决策树学习算法对付“过拟合”的主要手段，分为预剪枝和后剪枝

+ 预剪枝

    及早停止树的生长，选取属性“脐部”对训练集进行划分，以获得提升泛化性能的属性

+ 后剪枝

    先从训练集生成一颗完整决策树，再从底层结点向上逐层考察

这里在乳腺癌数据集上构造决策树，结果如下

![avator](F:\stuff\gitts\ML\pics\4_4_1.png)

可以看到在训练集上精度为1，说明叶结点都是纯的，出现过拟合，对新数据的泛化性能不佳，现在将预剪枝应用在决策树上，一种是在达到一定深度后停止树的展开，这里限制深度`max_depth=4`，这会降低训练集的精度

![avator](F:\stuff\gitts\ML\pics\4_4_2.png)

可以看到测试集精度明显提升

## 4.4 分析决策树

可以使用`tree`模块的`export_graphviz`函数来将树可视化，生成一个`.dot`文件，然后使用`graphviz`模块读取这个文件并将其可视化

```python
from sklearn.tree import export_graphviz
export_graphviz(tree, out_file="tree.dot", class_names=["malignant", "benign"], feature_names=dataset.feature_names, impurity=False, filled=True)
```

![avator](F:\stuff\gitts\ML\pythonProject\treePNG\tree.png)

## 4.5 树的特征重要性

查看整个树可能非常费劲，常用**特征重要性**总结树的工作原理，其为每个特征对树的决策的重要性进行排序

![avator](F:\stuff\gitts\ML\pics\4_5_1.png)

特征重要性始终为正数，特征和判定类别没有简单的关系，如以下数据集

![avator](F:\stuff\gitts\ML\pics\4_5_2.png)

对其进行决策树分析

![avator](F:\stuff\gitts\ML\pythonProject\treePNG\mytree.png)

回归树的用法和分析与分类树非常类似，基于树的回归模型**不能外推**，也不能在训练数据范围之外进行预测，这里以计算机内存历史价格数据集来说明

![avator](F:\stuff\gitts\ML\pics\4_5_3.png)

这里利用2000年以前的历史数据来预测2000年后的价格，只用日期来作为特征，对比决策树回归和线性回归

![avator](F:\stuff\gitts\ML\pics\4_5_4.png)

线性模型对测试数据给出了相当好的预测，不过忽略了训练数据和测试数据中一些更细微的变化；由于没有限制数的复杂度，树模型很好的预测了训练数据集，一旦超出了训练数据范围，树不能生成新的响应。如上所述，控制决策树模型复杂度的参数是预剪枝参数，即在树完全展开之前停止树的构造，选择一种预剪枝测量足以防止过拟合，泛化性能害是很差。

## 4.6 决策树集成

**集成**是合并多个机器学习模型来构造更强大模型的方法，**随机森林（random forest）**和**梯度提升决策树（gradient boosted decision tree）**都是以决策树为基础，对大量分类和回归的数据集都是有效的。

+ 随机森林

    随机森林本质上是许多决策树的集合，每棵树的预测可能都相对较好，但可能对部分数据过拟合，对树的结果取平均值来降低过拟合。在构造树过程中添加随机性以确保树各不相同，一种是*选择构造树的数据点随机*，一种是*每次划分测试随机*。

    首先对数据进行自助采样，即对一个数据集有放回的重复抽取一个样本，构造一个与原数据集大小相同的数据集，但会造成有些数据点的缺失以及重复；然后基于所构造的数据集来构造决策树，在每个结点处随机选择特征的一个子集，并对其中一个特征寻找最佳测试；对所有树的预测概率取平均值。下面将由5颗树组成的随机森林应用到`two_moons`数据集上，这里是取100个数据点

    数据集如下

    ![avator](F:\stuff\gitts\ML\pics\4_5_6.png)

    随机森林结果如下

    ![avator](F:\stuff\gitts\ML\pics\4_5_5.png)

    这里将包含100棵树的随机森林应用在乳腺癌数据集上

    ![avator](F:\stuff\gitts\ML\pics\4_5_7.png)

    其精度如下

    ![avator](F:\stuff\gitts\ML\pics\4_5_8.png)

    相比于决策树（单棵树），随机森林保留了更多特征，能从整体把握数据特征。**用于回归和分类**的随机森林是当前应用最广泛的机器学习方法之一，可是使用`n_jobs`参数调节使用的内核个数。对于维度非常高的稀疏数据，随机森林表现不是很好，

+ 梯度提升决策树

    梯度提升机，通过合并多个决策树来构造一个更为强大的模型，既可用于回归也可用于分类，采用连续的方式构造树，每棵树试图纠正前一棵树的错误，通常深度很小并对参数设置敏感，这里使用100棵树，最大深度为1，学习率设置为0.01，构建乳腺癌数据集模型

    ![avator](F:\stuff\gitts\ML\pics\4_5_9.png)

    可以看到，梯度提升完全忽略了某些特征。后续可研究`xgboost`包

# Chapter_5

## 5.1 神经元模型

神经网络是由具有适应性的简单单元组成的广泛并行互连的网络，其组织能够模拟生物神经系统对真实世界物体所作出的交互反应。在生物神经网络中，每个神经元与其他神经元相连，如果某神经元的电位超过一个“阈值”，那么就会被激活，向其他神经元发生化学物质

![avator](F:\stuff\gitts\ML\pics\5_1_1.png)

由于阶跃函数不现实，常用$Sigmoid$函数作为激活函数

![avator](F:\stuff\gitts\ML\pics\5_1_2.png)

将许多个神经元按一定层次结构连接起来就得到神经网络。深度学习模型是深层的神经网络，增加隐层数目或增加隐层神经元数目。深度学习算法往往经过精确调整，只适用于特定的使用场景。

## 5.2 感知机与多层网络

感知机由两层神经元组成，输入层接收外界输入信号后传递给输出层，输出层是$M-P$神经元，亦称“阈值逻辑单元”

![avator](F:\stuff\gitts\ML\pics\5_2_1.png)

更一般的，常见的神经网络是多层级结构，每层神经元与下一层神经元全互连，称“多层前馈神经网络”

![avator](F:\stuff\gitts\ML\pics\5_2_2.png)

**多层感知机**（multilayer perceptron, MLP)可作为研究更复杂深度学习的起点，也称为前馈神经网络。线性回归预测公式为
$$
\hat y = w[0]*x[0]+w[1]*x[1]+\cdots +w[p]*x[p]+b
$$
实质上是求输入特征$x[0]$到$x[p]$的加权拟合，权重为学习到的系数$w[0]$到$w[p]$，将这个公式可视化

![avator](F:\stuff\gitts\ML\pics\5_2_3.png)

左边每个结点代表一个输入特征，连线代表学到的系数；右边结点代表输出，是输入的加权求和。在MLP中多次重复这个计算加权求和的过程，首先计算代表中间过程的隐单元，然后计算这些隐单元的加权求和并得到最终结果

![avator](F:\stuff\gitts\ML\pics\5_2_4.png)

这个模型需要学习更多的系数（权重），在完成每个隐单元的加权求和之后，为提高模型拟合度，对结果再应用一个非线性函数（**校正非线性(relu)**或**正切双曲线(tanh)**），再将这个函数结果用于加权求和，计算输出$\hat y$，这两个函数可视化效果如下

![avator](F:\stuff\gitts\ML\pics\5_2_5.png)

relu截断小于0的值；而tanh在输入值较小时接近-1，在输入值较大时接近+1，这两种非线性函数的作用相当于滤波。

对于上面包含一层隐层的小型神经网络，计算回归问题$\hat y$的完整公式如下
$$
h[0]=tanh(w[0,0]*x[0]+w[1,0]*x[1]+w[2,0]*x[2]+w[3,0]*x[3]+b[0]) \\
h[1]=tanh(w[0,0]*x[0]+w[1,0]*x[1]+w[2,0]*x[2]+w[3,0]*x[3]+b[1]) \\
h[2]=tanh(w[0,0]*x[0]+w[1,0]*x[1]+w[2,0]*x[2]+w[3,0]*x[3]+b[2]) \\
\hat y =v[0]*h[0]+v[1]*h[1]+v[2]*h[2]+b
$$
其中$w$是输入层与隐层之间的权重，$v$是隐层与输出层之间的权重，权重$w$和$v$要从数据中学习得到，输出层与隐层间是线性关系，对于复杂数据，可添加多个隐层，以下是包含2个隐层，6个隐结点的小型神经网络

![avator](F:\stuff\gitts\ML\pics\5_2_6.png)

## 5.3 神经网络调参

由多层隐层组成的大型神经网络即是“深度学习”，这里将`MLPClassifier`应用到本章前的`two_moons`数据集上

![avator](F:\stuff\gitts\ML\pics\5_3_3.png)

可以看到决策边界完全是非线性的，但相对平滑。**MLP默认使用100个隐结点**，通过参数`hidden_layer_sizes`进行修改，这里改为10个

![avator](F:\stuff\gitts\ML\pics\5_3_4.png)

减少隐结点后，决策边界更加参差不齐，MLP默认使用relu非线性，如需实现更加平滑的决策边界，可添加**隐结点**、**添加隐层**或**使用tanh非线性**，使用2个隐层，每个包含10个单元

![avator](F:\stuff\gitts\ML\pics\5_3_5.png)

使用2个隐层，每个包含10个单元，使用tanh非线性

![avator](F:\stuff\gitts\ML\pics\5_3_6.png)

最后可使用L2惩罚使权重趋向于0，以控制神经网络复杂度，通过参数`alpha`调节，默认值很小（弱正则化），这里使用2个隐层，每个包含10个或100个单元，调节参数`alpha`进行观察

![avator](F:\stuff\gitts\ML\pics\5_3_7.png)

控制神经网络复杂度的方法有：隐层个数、每层单元个数、正则化参数等。神经网络在开始学习之前其权重是随机设置的，这种随机初始化会影响学到的模型，对小型网络会有所影响，以下几个模型都使用相同的参数设置进行学习

![avator](F:\stuff\gitts\ML\pics\5_3_8.png)

这里使用乳腺癌数据集，首先使用默认参数进行建模

![avator](F:\stuff\gitts\ML\pics\5_3_9.png)

可以看到结果与SVC差不多，原因可能在于数据的缩放，神经网络也要求所有输入特征的变化范围相似，最理想的情况是均值为0，方差为1，这里对数据集进行缩放

![avator](F:\stuff\gitts\ML\pics\5_3_10.png)

可以看到得到相当好的精度，但是给出了迭代次数达到最大的警告，应增加迭代次数，迭代次数默认为100，这里增加迭代次数，调节参数`max_iter`

![avator](F:\stuff\gitts\ML\pics\5_3_11.png)

可以看到在训练数据集上准确率达到了百分之百，虽然在测试集上准确率有所提升但存在过拟合的可能，这里增大正则化参数`alpha`，默认值是0.0001，向权重添加更强的正则化

![avator](F:\stuff\gitts\ML\pics\5_3_12.png)

可以看到在训练集上准确率有所降低，测试集上准确率达最佳。要想观察模型从训练集上学到了什么，即要想分析神经网络，一种是**查看模型权重**，对于乳腺癌数据集，这里给出该数据集30个输入特征与100个隐结点的对应关系，颜色深浅代表权重大小，通过热图展示

![avator](F:\stuff\gitts\ML\pics\5_3_13.png)

除了`sklearn`之外，害有`keras`、`lasagna`和`tensorflow`等深度学习库，`scikit-learn`不支持GPU加速。

误差逆传播（error BackPropagation，简称BP）算法可用于训练多层网络，将输出层误差逆向传播至隐层神经元，根据隐层神经元的误差来对连接权和阈值进行调整，其目的是最小化训练集上的累积误差

![avator](F:\stuff\gitts\ML\pics\5_3_1.png)

## 5.4 优缺点和参数

神经网络的训练过程可看作一个参数寻优过程，在参数空间中寻找一组最优参数使得**神经网络在训练集上的误差最小**。给定足够的计算时间和数据，并且仔细调节参数，NN将是最优算法；最重要的参数是隐层数和每层的隐单元（隐结点）个数，隐层数通常从1开始逐级递增，隐结点数通常与输入特征个数接近。

对于NN模型复杂度，一个有用的度量是**权重（系数）的个数**。常用调参方法是，首先创建一个大到足以过拟合的网络以确保该网络可以对任务进行学习，后续可缩小网络，亦可调节正则化参数，从而提高泛化性能。

在进行模型训练时，始终将数据缩放为均值为0、方差为1的数据集非常重要，主要关注点是：**隐层数**、**隐结点个数**、**正则化**和**非线性**。

## 5.5 常见网络

+ BRF网络

    径向基函数（Radial Basis Function）网络是一种单隐层前馈神经网络，使用径向基函数作为隐层神经元激活函数，输出层则是对隐层神经元输出的线性组合，具有足够多隐层神经元的BRF网络能以任意精度逼近任意连续函数

+ ART网络

    竞争型学习（competitive learning）是一种无监督学习策略，输出神经元相互竞争，每一时刻仅有一个竞争获胜的神经元被激活，其他神经元状态被抑制。自适应谐振理论（Adaptive Resonance Theory）网络是竞争型学习的代表

+ SOM网络

    自组织映射（Self-Organizing Map）网络是一种竞争学习型的无监督神经网络，将高维输入数据映射到低维空间，同时还能保持输入数据在高维空间的拓扑结构

+ 级联相关网络

    结构自适应网络则将网络结构也作为学习的目标之一，并希望能在训练过程中找到最符合数据特点的网络结构，级联相关网络即是结构自适应网络的代表

    ![avator](F:\stuff\gitts\ML\pics\5_3_2.png)

+ Elman网络

    递归神经网络允许网络中出现环形结构，即允许“反馈”，Elman网络是最常用的递归神经网络之一

    ![avator](F:\stuff\gitts\ML\pics\5_5_1.png)

+ Boltzmann机

    为网络状态定义一个“能量”，能量最小化时网络达到理想状态，Boltzmann机就是一种基于能量的模型

    ![avator](F:\stuff\gitts\ML\pics\5_5_2.png)

    将每个训练样本视为一个状态向量，使其出现的概率尽可能大。受限Boltzmann机常用“对比散度（Contrastive Divergence，简称CD）算法来进行训练

## 5.6 分类器的不确定度估计

分类器能给出预测的不确定度估计，即对预测的置信程度。大多数分类器有两个函数用于获取分类器的不确定估计：`decision_function`和`predict_proba`，这里构建一个`GradientBoostingClassifier`分类器，观察这两个函数对一个模拟的二维数据集`circles`的作用，`circles`数据集形状为（100，2），两个特征均为模拟量

![avator](F:\stuff\gitts\ML\pics\5_6_1.png)

### 5.6.3 二分类不确定度

+ 决策函数

对于二分类的情况，`descision_function`返回值的形状是`(n_samples,)`，为每个样本都返回一个浮点数，这个值可在任意范围内取值，即可进行任意缩放；这个值表示该模型对该数据点预测结果的置信程度，可通过查看决策函数的正负号来再现预测值，“反”类始终是`classes_`属性的第一个元素，“正”类是`classes_`的第二个元素，因此要再现`predict`的输出需要利用`classes_`属性，利用颜色编码画出所有点的`decision_function`和决策边界

![avator](F:\stuff\gitts\ML\pics\5_6_2.png)

这样即给出预测结果，又给出分类器的置信程度。

+ 预测概率

`predict_proba`输出的是每个类别的概率，对于二分类问题，其形状始终是`(n_samples, 2)`，每一行第一个元素是第一个类别的估计概率，第二个元素是第二个类别的估计概率，两个类别元素的估计概率之和始终为1，类别的估计概率即为置信程度，这里同样利用颜色编码画出所有点的`predict_proba`和决策边界

![avator](F:\stuff\gitts\ML\pics\5_6_3.png)

### 5.6.2 多分类问题不确定度

这里对`Iris`数据集这个三分类数据集进行分析

+ 决策函数

    对于多分类问题，决策函数的形状为`(n_samples, n_classes)`，每一列对应每个类别的“确定度分数”

+ 预测概率

    对于多分类问题，预测概率的形状为`(n_samples, n_classes)`，每一列对应每个类别的“确定度概率”


### 5.6.3 小结

+ 最邻近

    使用小型数据集

+ 线性模型

    适用于大型数据集

+ 朴素贝叶斯

    只适用于分类问题

+ 决策树

    不需要数据缩放，可以可视化

+ 随机森林

    不需要数据缩放，不适用于高维稀疏数据

+ 梯度递升决策树

    相比随机森林，需要更多参数调节

+ 支持向量机

    需要数据缩放，对参数敏感

+ 神经网络

    可以构建非常复杂网络，需要数据缩放，对参数敏感

# Chapter_6

## 6.1 间隔与核支持向量机

核支持向量机可推广到更复杂模型的扩展，这些模型无法被输入空间的超平面定义。给定训练集$D$，分类学习就是基于训练集$D$在样本空间中找到一个划分超平面，将不同类别的样本分开，通常这种划分超平面非常多

![avator](F:\stuff\gitts\ML\pics\6_1_1.png)

![avator](F:\stuff\gitts\ML\pics\6_1_2.png)

支持向量机（SVM）即是找到具有最大间隔的划分超平面

![avator](F:\stuff\gitts\ML\pics\6_1_3.png)

## 6.2 线性模型与非线性特征

线性模型在低维空间中可能非常受限，如以下数据，形状为(100, 2)，共有100个数据，每个数据有两个特征

![avator](F:\stuff\gitts\ML\pics\6_2_1.png)

用于分类的线性模型只能用一条直线来划分数据点，对这个数据集无法给出较好的结果

![avator](F:\stuff\gitts\ML\pics\6_2_2.png)

现对输入特征进行扩展，添加第二个特征的平凡作为一个新特征，将每个数据点表示为三维点`(feature0, feature1, feature1 ** 2)`，并作出其三维散点图

![avator](F:\stuff\gitts\ML\pics\6_2_3.png)

在三维空间中可用线性模型将这两个类别分开

![avator](F:\stuff\gitts\ML\pics\6_2_4.png)

如果将线性SVM模型看作原始特征的函数，实际上是非线性函数

![avator](F:\stuff\gitts\ML\pics\6_2_5.png)

## 6.3 核函数

对于训练样本是非线性可分的，可将样本从原始空间映射到更高维的特征空间，使其在此特征空间内线性可分

![avator](F:\stuff\gitts\ML\pics\6_3_1.png)

这里给出**核函数**的定义

![avator](F:\stuff\gitts\ML\pics\6_3_2.png)

对于支持向量机，将数据映射到更高维空间有两种常用方法

+ 多项式核

    在一定阶数内计算原始特征所有可能的多项式

+ 径向基函数

    也称高斯核，考虑所有阶数的所有可能的多项式

## 6.4 理解SVM

通常只有一部分训练数据点对于定义决策边界来说很重要，即位于类别之间边界上的点，这些点称为**支持向量**，基于这些点与决策边界之间距离确定决策边界，这里的距离表示数据点间的欧氏距离。下面在`handcraft`数据集（一个二维二分类数据集）上训练SVM，使用高斯核（rbf），支持向量是尺寸较大的点

![avator](F:\stuff\gitts\ML\pics\6_4_2.png)

## 6.5 SVM调参

`gamma`参数是用于控制高斯核的宽度，决定了点与点之间“靠近”是指多大的距离；`C`参数是正则化参数，用于限制点的重要性。这里可视化改变这些点的结果

![avator](F:\stuff\gitts\ML\pics\6_5_1.png)

`gamma`参数越小，高斯核半径就越大，点之间就被看作靠近，生成的模型复杂度就越小；`C`参数越小，说明模型非常受限，每个数据点的影响范围都有限，生成的模型复杂度就越小。将高斯核SVM应用到乳腺癌数据集上

![avator](F:\stuff\gitts\ML\pics\6_5_2.png)

SVM对参数的设定核数据的缩放非常敏感，要求所有特征有相似的变化范围，这里将每个特征的最值绘制在对数坐标上

![avator](F:\stuff\gitts\ML\pics\6_5_3.png)



## 6.6 SVM预处理数据

为解决特征具有完全不同的数量级这一问题，一种方法是对每个特征进行缩放到0和1之间；另一种方法就是调整参数，以改变模型复杂度。



# Chapter_7

## 7.1 无监督学习

数据集的**无监督变换**是创建数据新的表示的算法，**无监督变换**的一个常见应用是“降维”，用较少的特征概括数据；**聚类算法**将数据划分为不同的组，每组包含相似的物项。无监督学习的一个主要挑战就是**评估算法是否学到了有用的东西**。

## 7.2 预处理

一些算法对数据缩放非常敏感，通常做法是对特征进行调节，使数据表示更适合这些算法，即对数据的一种简单的按特征的缩放和移动

![avator](F:\stuff\gitts\ML\pics\7_2_1.png)

如图展示了四种数据变化方法

+ `StandardScaler`确保每个特征的平均值为0，方差为1，使所有特征位于同一量级
+ `RobustScaler`确保每个特征的统计属性都位于同一范围，使用中位数和四分位数
+ `MinMaxScaler`移动数据，使所有特征都刚好位于0到1之间
+ `Normalizer`对每个数据点进行缩放，使特征向量的欧式长度等于1，将数据点映射到半径为1的圆上，要求与特征向量的长度无关

这里使用乳腺癌数据集作为实例进行数据预处理，该数据集特征量级不统一，先使用`MinMaxScaler`进行数据预处理，`transform`方法总是对训练集和测试集应用完全相同的变换，会造成训练集和测试集变换最值的差异。对训练集和测试集应用完全相同的变换

![avator](F:\stuff\gitts\ML\pics\7_2_2.png)

图一展示的是原始数据集；图二展示的是`fit`作用在训练集上，然后调用`transform`作用在训练集和测试集上；图三展示的是对训练集和测试集分别进行缩放，特征的最值都是1和0，但是由于作了不同缩放改变了数据的排列。可以直接用`fit_transform`方法代替先`fit`后`transform`。

## 7.3 降维、特征提取与流形学习

数据变换的目的有**可视化**、**压缩数据**以及**进一步处理**等，最常用的是主成分分析，还有非负矩阵分解（NMF）和t-SNE。

### 7.3.1 主成分分析

主成分分析（principal component analysis，PCA）是一种旋转数据集的方法，旋转后的特征在统计上不相关。以下为PCA对一个模拟二维数据集的作用

![avator](F:\stuff\gitts\ML\pics\7_3_1.png)

算法首先找到方差最大的方向，将其标记为`Component 1`，这是数据中包含最多信息的方向



