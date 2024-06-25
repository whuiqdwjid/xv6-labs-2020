import numpy as np
# from sklearn.linear_model import LinearRegression

import matplotlib
matplotlib.use('Agg')  # 选择合适的后端，如Agg

import matplotlib.pyplot as plt

# 5个时刻的准确率数据

x = list(range(1,10))
y1 = list(range(11,20))
y2 = list(range(21,30))
plt.plot(x,y1,label='Benchmark value (prediction accuracy)')
plt.plot(x, y2, label='true accuracy')
print('ok')
plt.legend()
plt.title('Two Lines')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('./test.png')
print('ok')
plt.close()
# plt.show()
#
# def CreateLinearRegression(X, y):
#     model = LinearRegression()
#     model.fit(X, y)
#     predict_6th = model.predict([[len(X)+1]])
#     return len(X)+1,predict_6th
# 创建并拟合线性回归模型

# X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # 时刻
# y = np.array([0.8, 0.85, 0.9, 0.88, 0.92])   # 准确率
# num,z = CreateLinearRegression(X, y)
# print(num,z)
# 获取拟合后的截距和系数
# intercept = model.intercept_
# coef = model.coef_

# print("拟合后的函数式为:")
# # print(f"准确率 = {coef[0]:.3f} * 时刻 + {intercept:.3f}")
# #
# # # 预测第6时刻的准确率
# # predict_6th = model.predict([[6]])
# print("第6时刻的预测准确率:", predict_6th[0])
#
# # 绘制拟合曲线
# plt.scatter(X, y, color='blue', label='Actual Data')
# plt.plot(X, model.predict(X), color='red', label='Fitted Line')
# plt.xlabel('时刻')
# plt.ylabel('准确率')
# plt.title('拟合曲线')
# plt.legend()
# plt.show()