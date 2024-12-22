# sử dụng gradient descent để tìm cực tiểu của hàm f(x) = sin(x) * -x^2
# đạo hàm cuẩ f(x) = sin(x) * - x^2 là f'(x) = -2xsin(x) - x^2cos(x)
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return -x**2 * np.sin(x)

def derivative(x):
    return -2*x*np.sin(x) - x**2 * np.cos(x)


x = np.linspace(-10,10,500)
y = f(x)
plt.plot(x,y)
plt.xlabel('x')
plt.ylabel('y')
plt.axhline(0, color='black', linewidth=0.5)  # Đường ngang trục x
plt.axvline(0, color='black', linewidth=0.5)  # Đường dọc trục y
plt.grid(True)  # Thêm lưới

theta = 5
alpha = 0.001 #learning rate
eps = 0.00001

plt.scatter(theta, f(theta), color='red')
plt.pause(0.1)

v = derivative(theta)


cnt= 0
color_flag = True
p_color = 'red'
dampening = 0.5
points = []
while True:
    cnt += 1
    dtheta = derivative(theta)
    v = v * 0.99 + (1 - dampening) * dtheta
    theta = theta - alpha * v

    point, = plt.plot(theta, f(theta), 'o', color=p_color)
    points.append(point) 

    if len(points) > 5: 
        points[0].remove()  
        points.pop(0) 

    plt.pause(0.0167) 

    if cnt % 5 == 0:
        cnt = 0
        if color_flag:
            p_color = 'red'
        else:
            p_color = 'blue'
        color_flag = not color_flag

    if abs(dtheta) < eps:
        break

print("x= :", theta)