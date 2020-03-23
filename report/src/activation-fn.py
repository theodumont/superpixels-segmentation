# encoding: utf-8

import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
sb.set()


def sigmoid(x):
    a = []
    for item in x:
        a.append(1/(1+math.exp(-item)))
    return a


def tangh(x):
    return [math.tanh(item) for item in x]


def ReLU(x):
    return [max(0, item) for item in x]


def LReLU(x, alpha):
    return [max(alpha * item, item) for item in x]


x = np.arange(-10., 10., 0.2)
y = sigmoid(x)

fig, ax = plt.subplots()
ax.plot(x, y, color='m')
ax.set_xlim(-5, 5)
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.show()
