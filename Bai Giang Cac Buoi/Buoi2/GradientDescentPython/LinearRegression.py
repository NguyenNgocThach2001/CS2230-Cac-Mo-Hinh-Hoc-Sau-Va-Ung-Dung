import numpy as np
import matplotlib.pyplot as plt

# Fake data
def f(x):
    return 2 * x + 2

x_data = np.random.uniform(-5, 5, 20)
noise = np.random.normal(0, 1.5, 20)
y_data_noise = f(x_data) + noise
x_line = np.linspace(-5, 5, 200)

Y = y_data_noise.reshape(-1, 1)

X = np.column_stack([x_data, np.ones_like(x_data)])  
theta = np.array([0.0, 0.0]).reshape(-1, 1)  

def compute_gradients(X, Y, theta):
    m = len(Y)
    errors = X @ theta - Y
    grad0 = np.sum(errors) / m
    grad1 = np.sum(X[:, 0] * errors) / m
    return grad0, grad1

alpha = 0.001
eps = 0.001

plt.figure(figsize=(8, 6))
plt.scatter(x_data, y_data_noise, color='red', label='Dữ liệu nhiễu') 
plt.plot(x_line, f(x_line), label='Đường ban đầu (y=2x+2)', color='red')  
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True)
plt.title('Cập nhật đường thẳng dự đoán qua từng bước')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

line, = plt.plot(x_line, x_line * theta[1, 0] + theta[0, 0], color='green', label='Dự đoán')
v = [0, 0]
dampening = 0.7
while True:
    dt0, dt1 = compute_gradients(X, Y, theta)
    v[0] = v[0] * 0.99 + dt0 * (1-dampening)
    v[1] = v[1] * 0.99 + dt1 * (1-dampening)
    theta[0] = theta[0] - v[0] * alpha
    theta[1] = theta[1] - v[1] * alpha
    line.set_ydata(x_line * theta[1, 0] + theta[0, 0])
    plt.pause(0.016)  
    if abs(dt0) < eps and abs(dt1) < eps:
        break
    #print(dt0, dt1, theta[0, 0], theta[1, 0])
    
print("theta:", theta)
plt.show()
