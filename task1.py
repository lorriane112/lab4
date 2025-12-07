import pandas as pd
import matplotlib.pyplot as plt
import torch

### 加载鸢尾花数据集中的花萼长度和花萼宽度
def load_data():
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = pd.read_csv('iris.data.csv', names=names)
    X = dataset["sepal-length"].to_numpy()
    Y = dataset["sepal-width"].to_numpy()
    X = X.tolist()
    Y = Y.tolist()
    return X, Y


### 可视化X, Y的散点图，X、Y均为List，长度相同
def visualize_scatter(X, Y, xname, yname, title, w=None, b=None, color='red'):
    plt.scatter(X, Y)
    if(w is not None):
        y_line = w * torch.tensor(X) + b
        plt.plot(X, y_line, color=color)

    # 设置图表标题和坐标轴标签
    plt.title(title)
    plt.xlabel(xname)
    plt.ylabel(yname)

    # 显示图表
    plt.show()

    
### Task 1: 基于闭式解
def lr_cf(X, Y):
    w = 2
    b = 4
    ### TODO: 用线性回归的闭式解，计算参数w, b的估计值 (https://datawhalechina.github.io/thorough-pytorch/%E7%AC%AC%E4%BA%8C%E7%AB%A0/2.1%20%E5%BC%A0%E9%87%8F.html)
    return w, b

### Task 2: 基于梯度下降
def lr_gradient_descent(X, Y):
    w = 0.1
    b = 3
    ### TODO: 用梯度下降的方法，计算参数w, b的估计值（https://datawhalechina.github.io/thorough-pytorch/%E7%AC%AC%E4%BA%8C%E7%AB%A0/2.2%20%E8%87%AA%E5%8A%A8%E6%B1%82%E5%AF%BC.html）
    return w, b

    
### 执行程序
X, Y = load_data()
visualize_scatter(X, Y, 'Sepal Length', 'Sepal Width', 'IRIS Dataset')
w_cf, b_cf = lr_cf(X, Y)
visualize_scatter(X, Y, 'Sepal Length', 'Sepal Width', 'IRIS Dataset (Closed-Form)', w_cf, b_cf, color='blue')
w_gd, b_gd = lr_gradient_descent(X, Y)
visualize_scatter(X, Y, 'Sepal Length', 'Sepal Width', 'IRIS Dataset (Gradient-Descent)', w_gd, b_gd, color='red')