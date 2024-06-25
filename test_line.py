import numpy as np

import matplotlib
matplotlib.use('Agg')  # 选择合适的后端，如Agg

import matplotlib.pyplot as plt

from scipy.optimize import leastsq
from scipy.optimize import curve_fit
# from scipy.optimize import curve_fit

X = np.array([0,1,2,3])  # 时刻,1000
Y = np.array([0.0008000000000000229,0.000200000000000089,0.00019999999999997797,9.999999999998899e-05])   # 准确率,1
## 定义一次函数
def linear(x,a,b):
    return a * x * + b
## 定义二次函数
def quadratic(x,a,b,c):
    return a * x ** 2 + b * x + c
## 定义二次函数
def Cubic(x,a,b,c,d):
    return a * x ** 3 + b * x ** 2 + c*x + d
def logarithmic_func(x,a,b):
    # print(a,b,c)
    # print(x)
    return a*np.log(b * x+ 1e-10)
popt,pcov=curve_fit(f=logarithmic_func,xdata=X,ydata=Y,p0=[-1,1])#popt返回值是残差最小时的参数，即最佳参数 ,2.0
y_pred=[logarithmic_func(i,popt[0],popt[1]) for i in X]#将x和参数带进去，得到y的预测值 ,popt[2]
# print("y="+str(round(popt[0],2))+"x+"+str(round(popt[1],2))) # "+str(round(popt[1],2))+"
print(popt)
print(logarithmic_func(4,popt[0],popt[1])) #,popt[2]
plt.figure(figsize=(8, 6))
# X=np.delete(X,[5])
# Y=np.delete(Y,[5])
plt.scatter(X, Y, color="green", label="sample data", linewidth=2)
#   画 拟合直线
x = np.linspace(0, 10, 100)  ##在0-15直接画100个连续点
plt.plot(x, logarithmic_func(x, popt[0],popt[1]), 'r-', label='Fitted model')#,popt[2]
plt.legend()  # 绘制图例
plt.savefig('./Cubic_no_6th.png')
print('ok')
plt.close()


# # 对数线性模型，y = a*loge(x) + b /np.log(di)
# def log_linear_model(x, a, b):#, di
#     return a * (np.log(x)) + b
# # # 将数据点代入对数线性模型中，并取平方差作为代价函数
# # def cost_function(params, x, y):
# #     y_model = log_linear_model(x, *params)
# #     return ((y_model - y) ** 2).sum()
# #
# # # 初始参数
# # p0 = np.array([1.0, 1.0])
# # # 拟合
# # result = leastsq(cost_function, p0, args=(X,Y))
# # # 拟合结果
# # a, b = result[0]
# # print("y=" + str(round(a, 2)) + "loge(x)+" + str(round(b, 2)))
#
#
# popt,pcov=curve_fit(log_linear_model,X,Y,[1.0,1.0])#popt返回值是残差最小时的参数，即最佳参数 ,2.0
# y_pred=[log_linear_model(i,popt[0],popt[1]) for i in X]#将x和参数带进去，得到y的预测值 ,popt[2]
#
#
# print(popt)  # 输出参数
# print("y="+str(round(popt[0],2))+"log(x)+"+str(round(popt[1],2))) # "+str(round(popt[1],2))+"
# print(log_linear_model(6,popt[0],popt[1])) #,popt[2]
# plt.figure(figsize=(8, 6))
# # X=np.delete(X,[5])
# # Y=np.delete(Y,[5])
# plt.scatter(X, Y, color="green", label="sample data", linewidth=2)
#
# #   画拟合直线
# x = np.linspace(0, 15, 100)  ##在0-15直接画100个连续点
# plt.plot(x, log_linear_model(x, popt[0],popt[1]), 'r-', label='Fitted model')#,popt[2]
# plt.legend()  # 绘制图例
# plt.savefig('./curve_fit_e_di_no_6th.png')
# print('ok')
# plt.close()

# 获取拟合后的截距和系数
# intercept = model.intercept_
# coef = model.coef_

# print("拟合后的函数式为:")
# print(f"准确率 = {coef[0]:.3f} * 时刻 + {intercept:.3f}")
#
# # 预测第6时刻的准确率
# predict_6th = model.predict([[6]])
# print("第6时刻的预测准确率:", predict_6th[0])