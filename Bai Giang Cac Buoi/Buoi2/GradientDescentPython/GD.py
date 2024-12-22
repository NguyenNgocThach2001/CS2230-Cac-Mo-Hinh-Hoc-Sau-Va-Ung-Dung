# sử dụng gradient descent để tìm cực tiểu của hàm f(x) = 5x^2 - 3x + 7 
# đạo hàm cuẩ f(x) = 5x^2 - 3x + 7 là f'(x) = 10x - 3
import math
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return 5 * x**2 - 3 * x + 7  # Hàm f(x)

def derivative(theta):
    return 10 * theta - 3


x = np.linspace(-10,10,500)
y = f(x)
plt.plot(x,y)
plt.xlabel('x')
plt.ylabel('y')
plt.axhline(0, color='black', linewidth=0.5)  # Đường ngang trục x
plt.axvline(0, color='black', linewidth=0.5)  # Đường dọc trục y
plt.grid(True)  # Thêm lưới

theta = 5 
alpha = 0.01 #learning rate
eps = 0.1


plt.scatter(theta, f(theta), color='red')
while True:
    plt.scatter(theta, f(theta), color='red')
    dtheta = derivative(theta)
    theta = theta - alpha * dtheta
    plt.pause(0.1)
    if(abs(dtheta)<eps):
        break

print("x= :", theta)