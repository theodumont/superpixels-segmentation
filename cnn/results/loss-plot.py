"""
Training loss
"""
import numpy as np
import matplotlib.pyplot as plt

loss = []
for model in [3]:
    for idx in range(40):
        los = np.load("loss-train/run" + str(model) + "_" + str(idx) + ".npy")
        loss.append(np.mean(los))

validation_loss = np.load("loss-validation/Validation_loss_0.npy")
for model in []:
    validation_loss = np.concatenate((validation_loss, np.load("loss-validation/Validation_loss_" + str(model) + ".npy")))

plt.figure("Loss")
plt.semilogy(loss, label='Training loss')
plt.plot(validation_loss, label='Validation loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
