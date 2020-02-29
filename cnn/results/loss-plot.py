"""
Training loss
"""
import numpy as np
import matplotlib.pyplot as plt

loss = []
for model in [3, 4, 5, 0]:
    for idx in range(40):
        l = np.load("train/run" + str(model) + "_" + str(idx) + ".npy")
        loss.append(np.mean(l))

validation_loss = np.load("validation/Validation_loss_3.npy")
for model in [4, 5, 0]:
    validation_loss = np.concatenate((validation_loss, np.load("validation/Validation_loss_" + str(model) + ".npy")))

plt.figure()
plt.semilogy(loss, label='Training loss')
plt.semilogy(validation_loss, label='Validation loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()



