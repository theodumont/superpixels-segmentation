import math
import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    a = []
    for item in x:
        a.append(1/(1+math.exp(-item)))
    return a


def tangh(x):
    return [math.tanh(item) for item in x]


x = np.arange(-10., 10., 0.2)
fn = sigmoid(x)
fig, ax = plt.subplots()
ax.plot(x, fn)
ax.set_xlim(-5, 5)
plt.show()
